//=======- RawPtrRefMemberChecker.cpp ----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

class RawPtrRefMemberChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR;

protected:
  mutable std::optional<RetainTypeChecker> RTC;

public:
  RawPtrRefMemberChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool>
  isPtrCompatible(const clang::QualType,
                  const clang::CXXRecordDecl *R) const = 0;
  virtual bool isPtrCls(const clang::CXXRecordDecl *) const = 0;
  virtual const char *typeName() const = 0;
  virtual const char *invariant() const = 0;

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : ConstDynamicRecursiveASTVisitor {
      const RawPtrRefMemberChecker *Checker;
      explicit LocalVisitor(const RawPtrRefMemberChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool VisitTypedefDecl(const TypedefDecl *TD) override {
        if (Checker->RTC)
          Checker->RTC->visitTypedef(TD);
        return true;
      }

      bool VisitRecordDecl(const RecordDecl *RD) override {
        Checker->visitRecordDecl(RD);
        return true;
      }

      bool VisitObjCContainerDecl(const ObjCContainerDecl *CD) override {
        Checker->visitObjCDecl(CD);
        return true;
      }
    };

    LocalVisitor visitor(this);
    if (RTC)
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(TUD);
  }

  void visitRecordDecl(const RecordDecl *RD) const {
    if (shouldSkipDecl(RD))
      return;

    for (auto *Member : RD->fields()) {
      auto QT = Member->getType();
      const Type *MemberType = QT.getTypePtrOrNull();
      if (!MemberType)
        continue;

      if (auto *MemberCXXRD = MemberType->getPointeeCXXRecordDecl()) {
        std::optional<bool> IsCompatible = isPtrCompatible(QT, MemberCXXRD);
        if (IsCompatible && *IsCompatible)
          reportBug(Member, MemberType, MemberCXXRD, RD);
      } else {
        std::optional<bool> IsCompatible = isPtrCompatible(QT, nullptr);
        auto *PointeeType = MemberType->getPointeeType().getTypePtrOrNull();
        if (IsCompatible && *IsCompatible) {
          auto *Desugared = PointeeType->getUnqualifiedDesugaredType();
          if (auto *ObjCType = dyn_cast_or_null<ObjCInterfaceType>(Desugared))
            reportBug(Member, MemberType, ObjCType->getDecl(), RD);
        }
      }
    }
  }

  void visitObjCDecl(const ObjCContainerDecl *CD) const {
    if (auto *ID = dyn_cast<ObjCInterfaceDecl>(CD)) {
      for (auto *Ivar : ID->ivars())
        visitIvarDecl(CD, Ivar);
      return;
    }
    if (auto *ID = dyn_cast<ObjCImplementationDecl>(CD)) {
      for (auto *Ivar : ID->ivars())
        visitIvarDecl(CD, Ivar);
      return;
    }
  }

  void visitIvarDecl(const ObjCContainerDecl *CD,
                     const ObjCIvarDecl *Ivar) const {
    auto QT = Ivar->getType();
    const Type *IvarType = QT.getTypePtrOrNull();
    if (!IvarType)
      return;
    if (auto *IvarCXXRD = IvarType->getPointeeCXXRecordDecl()) {
      std::optional<bool> IsCompatible = isPtrCompatible(QT, IvarCXXRD);
      if (IsCompatible && *IsCompatible)
        reportBug(Ivar, IvarType, IvarCXXRD, CD);
    }
  }

  bool shouldSkipDecl(const RecordDecl *RD) const {
    if (!RD->isThisDeclarationADefinition())
      return true;

    if (RD->isImplicit())
      return true;

    if (RD->isLambda())
      return true;

    // If the construct doesn't have a source file, then it's not something
    // we want to diagnose.
    const auto RDLocation = RD->getLocation();
    if (!RDLocation.isValid())
      return true;

    const auto Kind = RD->getTagKind();
    // FIMXE: Should we check union members too?
    if (Kind != TagTypeKind::Struct && Kind != TagTypeKind::Class)
      return true;

    // Ignore CXXRecords that come from system headers.
    if (BR->getSourceManager().isInSystemHeader(RDLocation))
      return true;

    // Ref-counted smartpointers actually have raw-pointer to uncounted type as
    // a member but we trust them to handle it correctly.
    auto CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(RD);
    if (CXXRD)
      return isPtrCls(CXXRD);

    return false;
  }

  template <typename DeclType, typename PointeeType, typename ParentDeclType>
  void reportBug(const DeclType *Member, const Type *MemberType,
                 const PointeeType *Pointee,
                 const ParentDeclType *ClassCXXRD) const {
    assert(Member);
    assert(MemberType);
    assert(Pointee);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (isa<ObjCContainerDecl>(ClassCXXRD))
      Os << "Instance variable ";
    else
      Os << "Member variable ";
    printQuotedName(Os, Member);
    Os << " in ";
    printQuotedQualifiedName(Os, ClassCXXRD);
    Os << " is a ";
    if (printPointer(Os, MemberType) == PrintDeclKind::Pointer) {
      auto Typedef = MemberType->getAs<TypedefType>();
      assert(Typedef);
      printQuotedQualifiedName(Os, Typedef->getDecl());
    } else
      printQuotedQualifiedName(Os, Pointee);
    Os << "; " << invariant() << ".";

    PathDiagnosticLocation BSLoc(Member->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Member->getSourceRange());
    BR->emitReport(std::move(Report));
  }

  enum class PrintDeclKind { Pointee, Pointer };
  virtual PrintDeclKind printPointer(llvm::raw_svector_ostream &Os,
                                     const Type *T) const {
    T = T->getUnqualifiedDesugaredType();
    bool IsPtr = isa<PointerType>(T) || isa<ObjCObjectPointerType>(T);
    Os << (IsPtr ? "raw pointer" : "reference") << " to " << typeName() << " ";
    return PrintDeclKind::Pointee;
  }
};

class NoUncountedMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncountedMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "reference-countable type") {}

  std::optional<bool>
  isPtrCompatible(const clang::QualType,
                  const clang::CXXRecordDecl *R) const final {
    return R && isRefCountable(R);
  }

  bool isPtrCls(const clang::CXXRecordDecl *R) const final {
    return isRefCounted(R);
  }

  const char *typeName() const final { return "ref-countable type"; }

  const char *invariant() const final {
    return "member variables must be Ref, RefPtr, WeakRef, or WeakPtr";
  }
};

class NoUncheckedPtrMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncheckedPtrMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "checked-pointer capable type") {}

  std::optional<bool>
  isPtrCompatible(const clang::QualType,
                  const clang::CXXRecordDecl *R) const final {
    return R && isCheckedPtrCapable(R);
  }

  bool isPtrCls(const clang::CXXRecordDecl *R) const final {
    return isCheckedPtr(R);
  }

  const char *typeName() const final { return "CheckedPtr capable type"; }

  const char *invariant() const final {
    return "member variables must be a CheckedPtr, CheckedRef, WeakRef, or "
           "WeakPtr";
  }
};

class NoUnretainedMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUnretainedMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "retainable type") {
    RTC = RetainTypeChecker();
  }

  std::optional<bool>
  isPtrCompatible(const clang::QualType QT,
                  const clang::CXXRecordDecl *) const final {
    return RTC->isUnretained(QT);
  }

  bool isPtrCls(const clang::CXXRecordDecl *R) const final {
    return isRetainPtr(R);
  }

  const char *typeName() const final { return "retainable type"; }

  const char *invariant() const final {
    return "member variables must be a RetainPtr";
  }

  PrintDeclKind printPointer(llvm::raw_svector_ostream &Os,
                             const Type *T) const final {
    if (!isa<ObjCObjectPointerType>(T) && T->getAs<TypedefType>()) {
      Os << typeName() << " ";
      return PrintDeclKind::Pointer;
    }
    return RawPtrRefMemberChecker::printPointer(Os, T);
  }
};

} // namespace

void ento::registerNoUncountedMemberChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoUncountedMemberChecker>();
}

bool ento::shouldRegisterNoUncountedMemberChecker(const CheckerManager &Mgr) {
  return true;
}

void ento::registerNoUncheckedPtrMemberChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoUncheckedPtrMemberChecker>();
}

bool ento::shouldRegisterNoUncheckedPtrMemberChecker(
    const CheckerManager &Mgr) {
  return true;
}

void ento::registerNoUnretainedMemberChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoUnretainedMemberChecker>();
}

bool ento::shouldRegisterNoUnretainedMemberChecker(const CheckerManager &Mgr) {
  return true;
}
