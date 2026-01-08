//=======- RawPtrRefMemberChecker.cpp ----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
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
  mutable llvm::DenseSet<const ObjCIvarDecl *> IvarDeclsToIgnore;

protected:
  mutable std::optional<RetainTypeChecker> RTC;

public:
  RawPtrRefMemberChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool> isUnsafePtr(QualType,
                                          bool ignoreARC = false) const = 0;
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

    for (auto *Member : RD->fields())
      visitMember(Member, RD);
  }

  void visitMember(const FieldDecl *Member, const RecordDecl *RD) const {
    auto QT = Member->getType();
    const Type *MemberType = QT.getTypePtrOrNull();

    while (MemberType) {
      auto IsUnsafePtr = isUnsafePtr(QT);
      if (IsUnsafePtr && *IsUnsafePtr)
        break;
      if (!MemberType->isPointerType())
        return;
      QT = MemberType->getPointeeType();
      MemberType = QT.getTypePtrOrNull();
    }

    if (!MemberType)
      return;

    if (auto *MemberCXXRD = MemberType->getPointeeCXXRecordDecl())
      reportBug(Member, MemberType, MemberCXXRD, RD);
    else if (auto *ObjCDecl = getObjCDecl(MemberType))
      reportBug(Member, MemberType, ObjCDecl, RD);
  }

  ObjCInterfaceDecl *getObjCDecl(const Type *TypePtr) const {
    auto *PointeeType = TypePtr->getPointeeType().getTypePtrOrNull();
    if (!PointeeType)
      return nullptr;
    auto *Desugared = PointeeType->getUnqualifiedDesugaredType();
    if (!Desugared)
      return nullptr;
    if (auto *ObjCType = dyn_cast<ObjCInterfaceType>(Desugared))
      return ObjCType->getDecl();
    if (auto *ObjCType = dyn_cast<ObjCObjectType>(Desugared))
      return ObjCType->getInterface();
    return nullptr;
  }

  void visitObjCDecl(const ObjCContainerDecl *CD) const {
    if (BR->getSourceManager().isInSystemHeader(CD->getLocation()))
      return;

    if (auto *ID = dyn_cast<ObjCImplementationDecl>(CD)) {
      ObjCContainerDecl::PropertyMap map;
      CD->collectPropertiesToImplement(map);
      for (auto it : map)
        visitObjCPropertyDecl(CD, it.second);

      if (auto *Interface = ID->getClassInterface()) {
        for (auto *Ivar : Interface->ivars())
          visitIvarDecl(CD, Ivar);
      }
      for (auto *PropImpl : ID->property_impls())
        visitPropImpl(CD, PropImpl);
      for (auto *Ivar : ID->ivars())
        visitIvarDecl(CD, Ivar);
      return;
    }
  }

  void visitIvarDecl(const ObjCContainerDecl *CD,
                     const ObjCIvarDecl *Ivar) const {
    if (BR->getSourceManager().isInSystemHeader(Ivar->getLocation()))
      return;

    if (IvarDeclsToIgnore.contains(Ivar))
      return;

    auto QT = Ivar->getType();
    const Type *IvarType = QT.getTypePtrOrNull();
    if (!IvarType)
      return;

    auto IsUnsafePtr = isUnsafePtr(QT);
    if (!IsUnsafePtr || !*IsUnsafePtr)
      return;

    IvarDeclsToIgnore.insert(Ivar);

    if (auto *MemberCXXRD = IvarType->getPointeeCXXRecordDecl())
      reportBug(Ivar, IvarType, MemberCXXRD, CD);
    else if (auto *ObjCDecl = getObjCDecl(IvarType))
      reportBug(Ivar, IvarType, ObjCDecl, CD);
  }

  void visitObjCPropertyDecl(const ObjCContainerDecl *CD,
                             const ObjCPropertyDecl *PD) const {
    if (BR->getSourceManager().isInSystemHeader(PD->getLocation()))
      return;

    if (const ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(CD)) {
      if (!RTC || !RTC->defaultSynthProperties() ||
          ID->isObjCRequiresPropertyDefs())
        return;
    }

    auto [IsUnsafe, PropType] = isPropImplUnsafePtr(PD);
    if (!IsUnsafe)
      return;

    if (auto *MemberCXXRD = PropType->getPointeeCXXRecordDecl())
      reportBug(PD, PropType, MemberCXXRD, CD);
    else if (auto *ObjCDecl = getObjCDecl(PropType))
      reportBug(PD, PropType, ObjCDecl, CD);
  }

  void visitPropImpl(const ObjCContainerDecl *CD,
                     const ObjCPropertyImplDecl *PID) const {
    if (BR->getSourceManager().isInSystemHeader(PID->getLocation()))
      return;

    if (PID->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      return;

    auto *PropDecl = PID->getPropertyDecl();
    if (auto *IvarDecl = PID->getPropertyIvarDecl()) {
      if (IvarDeclsToIgnore.contains(IvarDecl))
        return;
      IvarDeclsToIgnore.insert(IvarDecl);
    }
    auto [IsUnsafe, PropType] = isPropImplUnsafePtr(PropDecl);
    if (!IsUnsafe)
      return;

    if (auto *MemberCXXRD = PropType->getPointeeCXXRecordDecl())
      reportBug(PropDecl, PropType, MemberCXXRD, CD);
    else if (auto *ObjCDecl = getObjCDecl(PropType))
      reportBug(PropDecl, PropType, ObjCDecl, CD);
  }

  std::pair<bool, const Type *>
  isPropImplUnsafePtr(const ObjCPropertyDecl *PD) const {
    if (!PD)
      return {false, nullptr};

    auto QT = PD->getType();
    const Type *PropType = QT.getTypePtrOrNull();
    if (!PropType)
      return {false, nullptr};

    // "assign" property doesn't retain even under ARC so treat it as unsafe.
    bool ignoreARC =
        !PD->isReadOnly() && PD->getSetterKind() == ObjCPropertyDecl::Assign;
    auto IsUnsafePtr = isUnsafePtr(QT, ignoreARC);
    return {IsUnsafePtr && *IsUnsafePtr && !PD->isRetaining(), PropType};
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
    if (Kind != TagTypeKind::Struct && Kind != TagTypeKind::Class &&
        Kind != TagTypeKind::Union)
      return true;

    // Ignore CXXRecords that come from system headers.
    if (BR->getSourceManager().isInSystemHeader(RDLocation))
      return true;

    // Ref-counted smartpointers actually have raw-pointer to uncounted type as
    // a member but we trust them to handle it correctly.
    auto CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(RD);
    if (CXXRD && isSmartPtr(CXXRD))
      return true;

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

    if (isa<ObjCContainerDecl>(ClassCXXRD)) {
      if (isa<ObjCPropertyDecl>(Member))
        Os << "Property ";
      else
        Os << "Instance variable ";
    } else
      Os << "Member variable ";
    printQuotedName(Os, Member);
    Os << " in ";
    printQuotedQualifiedName(Os, ClassCXXRD);
    if (Member->getType().getTypePtrOrNull() == MemberType)
      Os << " is a ";
    else
      Os << " contains a ";
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

  std::optional<bool> isUnsafePtr(QualType QT, bool) const final {
    return isUncountedPtr(QT.getCanonicalType());
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

  std::optional<bool> isUnsafePtr(QualType QT, bool) const final {
    return isUncheckedPtr(QT.getCanonicalType());
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

  std::optional<bool> isUnsafePtr(QualType QT, bool ignoreARC) const final {
    return RTC->isUnretained(QT, ignoreARC);
  }

  const char *typeName() const final { return "retainable type"; }

  const char *invariant() const final {
    return "member variables must be a RetainPtr or OSObjectPtr";
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
