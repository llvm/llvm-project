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
  // Maps a template instantiation to the enclosing template instantiation
  // whose instantiation caused it.
  mutable llvm::DenseMap<const Decl *, const Decl *> InstantiationCauses;
  // Template instantiations referenced by template definitions themselves.
  mutable llvm::DenseSet<const Decl *> NonDependentUses;

protected:
  const std::unique_ptr<PtrRefSafetyModel> Model;

public:
  RawPtrRefMemberChecker(const char *description,
                         std::unique_ptr<PtrRefSafetyModel> Model)
      : Bug(this, description, "WebKit coding guidelines"),
        Model(std::move(Model)) {}

  std::optional<bool> isUnsafePtr(QualType QT, bool IgnoreARC = false) const {
    return isUnsafePtrForStorage(*Model, QT, IgnoreARC);
  }

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;
    InstantiationCauses.clear();
    NonDependentUses.clear();

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : ConstDynamicRecursiveASTVisitor {
      const RawPtrRefMemberChecker *Checker;
      // Template instantiations are checked after the traversal so that we
      // know which enclosing template instantiation caused each of them.
      llvm::SmallVector<const CXXRecordDecl *> Instantiations;
      // The enclosing classes and functions of the node being visited.
      llvm::SmallVector<const Decl *> Contexts;

      explicit LocalVisitor(const RawPtrRefMemberChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool TraverseDecl(const Decl *D) override {
        bool IsContext = isa_and_nonnull<CXXRecordDecl, FunctionDecl>(D);
        if (IsContext)
          Contexts.push_back(D);
        bool Result = ConstDynamicRecursiveASTVisitor::TraverseDecl(D);
        if (IsContext)
          Contexts.pop_back();
        return Result;
      }

      bool VisitTypedefDecl(const TypedefDecl *TD) override {
        if (auto *RTC = Checker->Model->retainTypeChecker())
          RTC->visitTypedef(TD);
        return true;
      }

      bool VisitRecordDecl(const RecordDecl *RD) override {
        auto *CXXRD = dyn_cast<CXXRecordDecl>(RD);
        if (CXXRD && getEnclosingInstantiation(CXXRD))
          Instantiations.push_back(CXXRD);
        else
          Checker->visitRecordDecl(RD);
        return true;
      }

      bool VisitObjCContainerDecl(const ObjCContainerDecl *CD) override {
        Checker->visitObjCDecl(CD);
        return true;
      }

      bool VisitTypeLoc(TypeLoc TL) override {
        if (!TL.getType().isNull())
          Checker->recordUse(TL.getType()->getAsCXXRecordDecl(), Contexts);
        return true;
      }

      bool VisitDeclRefExpr(const DeclRefExpr *E) override {
        Checker->recordUse(E->getDecl(), Contexts);
        return true;
      }

      bool VisitMemberExpr(const MemberExpr *E) override {
        Checker->recordUse(E->getMemberDecl(), Contexts);
        return true;
      }
    };

    LocalVisitor visitor(this);
    if (auto *RTC = Model->retainTypeChecker())
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(TUD);
    for (auto *RD : visitor.Instantiations)
      visitRecordDecl(RD);
  }

  void visitRecordDecl(const RecordDecl *RD) const {
    if (shouldSkipDecl(RD))
      return;

    const CXXRecordDecl *Pattern = nullptr;
    SourceLocation InstantiationLoc;
    if (auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      if (auto *Instantiation = getEnclosingInstantiation(CXXRD)) {
        Pattern = Instantiation->getTemplateInstantiationPattern();
        InstantiationLoc = getReportLocation(Instantiation);
      }
    }

    for (auto *Member : RD->fields()) {
      if (Pattern) {
        // Members whose type doesn't depend on template parameters are
        // diagnosed in the template definition itself.
        if (!isDependentInPattern(Member, Pattern))
          continue;
        if (InstantiationLoc.isValid()) {
          visitMember(Member, RD, InstantiationLoc);
          continue;
        }
      }
      visitMember(Member, RD);
    }
  }

  static bool isDependentInPattern(const FieldDecl *Member,
                                   const CXXRecordDecl *Pattern) {
    if (!Pattern || !Member->getDeclName())
      return true;
    // Members of anonymous structs and unions are found in the enclosing
    // record as IndirectFieldDecls.
    for (auto *Found : Pattern->lookup(Member->getDeclName())) {
      const FieldDecl *PatternField = dyn_cast<FieldDecl>(Found);
      if (auto *Indirect = dyn_cast<IndirectFieldDecl>(Found))
        PatternField = Indirect->getAnonField();
      if (PatternField)
        return PatternField->getType()->isDependentType();
    }
    return true;
  }

  // Returns RD if it's a template instantiation. Anonymous structs and unions
  // don't have their own template specialization kind, so returns their
  // enclosing record if it's a template instantiation.
  static const CXXRecordDecl *
  getEnclosingInstantiation(const CXXRecordDecl *RD) {
    while (RD && RD->isAnonymousStructOrUnion())
      RD = dyn_cast<CXXRecordDecl>(RD->getDeclContext());
    return RD && isInstantiation(RD) ? RD : nullptr;
  }

  static bool isInstantiation(const Decl *D) {
    if (auto *RD = dyn_cast<CXXRecordDecl>(D))
      return isTemplateInstantiation(RD->getTemplateSpecializationKind());
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return isTemplateInstantiation(FD->getTemplateSpecializationKind());
    return false;
  }

  static const Decl *getInstantiationPattern(const Decl *D) {
    if (auto *RD = dyn_cast<CXXRecordDecl>(D))
      return RD->getTemplateInstantiationPattern();
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return FD->getTemplateInstantiationPattern();
    return nullptr;
  }

  static SourceLocation getPointOfInstantiation(const Decl *D) {
    if (auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(D))
      return Spec->getPointOfInstantiation();
    if (auto *RD = dyn_cast<CXXRecordDecl>(D)) {
      if (auto *Info = RD->getMemberSpecializationInfo())
        return Info->getPointOfInstantiation();
    }
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return FD->getPointOfInstantiation();
    return SourceLocation();
  }

  // Records that Used is referenced from the innermost template instantiation
  // or template definition in Contexts.
  void recordUse(const Decl *Used, ArrayRef<const Decl *> Contexts) const {
    if (!Used || !isInstantiation(Used))
      return;
    for (auto *Context : llvm::reverse(Contexts)) {
      if (Context == Used)
        return;
      if (isInstantiation(Context)) {
        // If Used was instantiated from within the definition of the template
        // Context was instantiated from, Context caused its instantiation.
        auto *Pattern = getInstantiationPattern(Context);
        auto &SM = BR->getSourceManager();
        auto Loc = getPointOfInstantiation(Used);
        if (Pattern && Loc.isValid() &&
            SM.isPointWithin(Loc, Pattern->getBeginLoc(), Pattern->getEndLoc()))
          InstantiationCauses.try_emplace(Used, Context);
        return;
      }
      if (cast<DeclContext>(Context)->isDependentContext()) {
        // The template definition itself refers to this instantiation.
        NonDependentUses.insert(Used);
        return;
      }
    }
  }

  // Returns where the outermost template instantiation which (transitively)
  // caused D to be instantiated is instantiated.
  SourceLocation getReportLocation(const Decl *D) const {
    llvm::SmallPtrSet<const Decl *, 4> Visited;
    while (Visited.insert(D).second && !NonDependentUses.contains(D)) {
      auto It = InstantiationCauses.find(D);
      if (It == InstantiationCauses.end())
        break;
      D = It->second;
    }
    return getPointOfInstantiation(D);
  }

  void visitMember(const FieldDecl *Member, const RecordDecl *RD,
                   SourceLocation ReportLoc = SourceLocation()) const {
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
      reportBug(Member, MemberType, MemberCXXRD, RD, ReportLoc);
    else if (auto *ObjCDecl = getObjCDeclFromObjCPtr(MemberType))
      reportBug(Member, MemberType, ObjCDecl, RD, ReportLoc);
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
    else if (auto *ObjCDecl = getObjCDeclFromObjCPtr(IvarType))
      reportBug(Ivar, IvarType, ObjCDecl, CD);
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

    if (auto *MemberCXXRD = PropType->getPointeeCXXRecordDecl())
      reportBug(PD, PropType, MemberCXXRD, CD);
    else if (auto *ObjCDecl = getObjCDeclFromObjCPtr(PropType))
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
    else if (auto *ObjCDecl = getObjCDeclFromObjCPtr(PropType))
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

    return false;
  }

  template <typename DeclType, typename PointeeType, typename ParentDeclType>
  void reportBug(const DeclType *Member, const Type *MemberType,
                 const PointeeType *Pointee, const ParentDeclType *ClassCXXRD,
                 SourceLocation ReportLoc = SourceLocation()) const {
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
    Os << " (of ";
    printQuotedQualifiedName(Os, ClassCXXRD);
    Os << ")";
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

    // Members of template instantiations are reported where the template is
    // instantiated since the template definition itself may be fine.
    bool IsInstantiation = ReportLoc.isValid();
    if (!IsInstantiation)
      ReportLoc = Member->getSourceRange().getBegin();
    PathDiagnosticLocation BSLoc(ReportLoc, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    if (!IsInstantiation)
      Report->addRange(Member->getSourceRange());
    if (auto *EnclosingCtx = Member->getDeclContext())
      Report->setDeclWithIssue(Decl::castFromDeclContext(EnclosingCtx));
    BR->emitReport(std::move(Report));
  }

  enum class PrintDeclKind { Pointee, Pointer };
  PrintDeclKind printPointer(llvm::raw_svector_ostream &Os,
                             const Type *T) const {
    // Retain/OS types are frequently spelled through a typedef (e.g. CFXXXRef);
    // print the typedef name rather than desugaring to the pointee.
    if (Model->retainTypeChecker() && !isa<ObjCObjectPointerType>(T) &&
        T->getAs<TypedefType>()) {
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
