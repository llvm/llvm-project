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
#include "RawPtrRefSafetyModel.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
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
  const std::unique_ptr<PtrRefSafetyModel> Model;

public:
  RawPtrRefMemberChecker(const char *description,
                         std::unique_ptr<PtrRefSafetyModel> Model)
      : Bug(this, description, "WebKit coding guidelines"),
        Model(std::move(Model)) {}

  std::optional<bool> isUnsafePtr(QualType QT, bool IgnoreARC = false) const {
    // Retain and release don't affect a class object's lifetime, even one
    // created at runtime, so a RetainPtr can't make a Class member safer.
    if (!QT.isNull() &&
        (QT->isObjCClassType() || QT->isObjCQualifiedClassType()))
      return false;
    return isUnsafePtrForStorage(*Model, QT, IgnoreARC);
  }

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
        if (auto *RTC = Checker->Model->retainTypeChecker())
          RTC->visitTypedef(TD);
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
    if (auto *RTC = Model->retainTypeChecker())
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

    reportBug(Member, MemberType, RD);
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

    reportBug(Ivar, IvarType, CD);
  }

  void visitObjCPropertyDecl(const ObjCContainerDecl *CD,
                             const ObjCPropertyDecl *PD) const {
    if (BR->getSourceManager().isInSystemHeader(PD->getLocation()))
      return;

    if (const ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(CD)) {
      auto *RTC = Model->retainTypeChecker();
      if (!RTC || !RTC->defaultSynthProperties() ||
          ID->isObjCRequiresPropertyDefs())
        return;
    }

    auto [IsUnsafe, PropType] = isPropImplUnsafePtr(PD);
    if (!IsUnsafe)
      return;

    reportBug(PD, PropType, CD);
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

    reportBug(PropDecl, PropType, CD);
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
    bool IsWeak =
        PD->getPropertyAttributes() & ObjCPropertyAttribute::kind_weak;
    bool HasSafeAttr = PD->isRetaining() || IsWeak;
    auto IsUnsafePtr = isUnsafePtr(QT, ignoreARC);
    return {IsUnsafePtr && *IsUnsafePtr && !HasSafeAttr, PropType};
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
    if (auto *Tmpl = CXXRD ? CXXRD->getDescribedClassTemplate() : nullptr) {
      if (isSmartPtrClass(safeGetName(Tmpl)))
        return true;
    }

    return false;
  }

  static const NamedDecl *getPointeeDecl(const Type *T) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl())
      return CXXRD;
    return getObjCDeclFromObjCPtr(T);
  }

  template <typename DeclType, typename ParentDeclType>
  void reportBug(const DeclType *Member, const Type *MemberType,
                 const ParentDeclType *ClassCXXRD) const {
    assert(Member);
    assert(MemberType);

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
    Os << " (of ";
    printQuotedQualifiedName(Os, ClassCXXRD);
    Os << ")";
    if (Member->getType().getTypePtrOrNull() == MemberType)
      Os << " is a ";
    else
      Os << " contains a ";
    if (printPointer(Os, MemberType) == PrintDeclKind::Pointer) {
      if (auto *Typedef = MemberType->getAs<TypedefType>())
        printQuotedQualifiedName(Os, Typedef->getDecl());
      else {
        Os << "'";
        QualType(MemberType, 0)
            .print(Os, Member->getASTContext().getPrintingPolicy());
        Os << "'";
      }
    } else
      printQuotedQualifiedName(Os, getPointeeDecl(MemberType));

    PathDiagnosticLocation BSLoc(Member->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Member->getSourceRange());
    if (auto *EnclosingCtx = Member->getDeclContext())
      Report->setDeclWithIssue(Decl::castFromDeclContext(EnclosingCtx));
    BR->emitReport(std::move(Report));
  }

  enum class PrintDeclKind { Pointee, Pointer };
  PrintDeclKind printPointer(llvm::raw_svector_ostream &Os,
                             const Type *T) const {
    // Retain/OS types are frequently spelled through a typedef (e.g. CFXXXRef),
    // and id<P> has no pointee decl; name the pointer type itself for both.
    bool IsRetainTypedef = Model->retainTypeChecker() &&
                           !isa<ObjCObjectPointerType>(T) &&
                           T->getAs<TypedefType>();
    if (IsRetainTypedef || !getPointeeDecl(T)) {
      Os << Model->typeName() << " ";
      return PrintDeclKind::Pointer;
    }
    T = T->getUnqualifiedDesugaredType();
    bool IsPtr = isa<PointerType>(T) || isa<ObjCObjectPointerType>(T);
    Os << (IsPtr ? "raw pointer" : "reference") << " to " << Model->typeName()
       << " ";
    return PrintDeclKind::Pointee;
  }
};

class NoUncountedMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncountedMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "reference-countable type",
                               makeRefPtrSafetyModel()) {}
};

class NoUncheckedPtrMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncheckedPtrMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "checked-pointer capable type",
                               makeCheckedPtrSafetyModel()) {}
};

class NoUnretainedMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUnretainedMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "retainable type",
                               makeRetainPtrSafetyModel()) {}
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
