//===--- FindTarget.cpp - What does an AST node refer to? -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindTarget.h"
#include "AST.h"
#include "support/Logger.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {
using ast_type_traits::DynTypedNode;

LLVM_ATTRIBUTE_UNUSED std::string
nodeToString(const ast_type_traits::DynTypedNode &N) {
  std::string S = std::string(N.getNodeKind().asStringRef());
  {
    llvm::raw_string_ostream OS(S);
    OS << ": ";
    N.print(OS, PrintingPolicy(LangOptions()));
  }
  std::replace(S.begin(), S.end(), '\n', ' ');
  return S;
}

// Helper function for getMembersReferencedViaDependentName()
// which takes a possibly-dependent type `T` and heuristically
// resolves it to a CXXRecordDecl in which we can try name lookup.
CXXRecordDecl *resolveTypeToRecordDecl(const Type *T) {
  assert(T);

  if (const auto *RT = T->getAs<RecordType>())
    return dyn_cast<CXXRecordDecl>(RT->getDecl());

  if (const auto *ICNT = T->getAs<InjectedClassNameType>())
    T = ICNT->getInjectedSpecializationType().getTypePtrOrNull();
  if (!T)
    return nullptr;

  const auto *TST = T->getAs<TemplateSpecializationType>();
  if (!TST)
    return nullptr;

  const ClassTemplateDecl *TD = dyn_cast_or_null<ClassTemplateDecl>(
      TST->getTemplateName().getAsTemplateDecl());
  if (!TD)
    return nullptr;

  return TD->getTemplatedDecl();
}

// Given a tag-decl type and a member name, heuristically resolve the
// name to one or more declarations.
// The current heuristic is simply to look up the name in the primary
// template. This is a heuristic because the template could potentially
// have specializations that declare different members.
// Multiple declarations could be returned if the name is overloaded
// (e.g. an overloaded method in the primary template).
// This heuristic will give the desired answer in many cases, e.g.
// for a call to vector<T>::size().
// The name to look up is provided in the form of a factory that takes
// an ASTContext, because an ASTContext may be needed to obtain the
// name (e.g. if it's an operator name), but the caller may not have
// access to an ASTContext.
std::vector<const NamedDecl *> getMembersReferencedViaDependentName(
    const Type *T,
    llvm::function_ref<DeclarationName(ASTContext &)> NameFactory,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
  if (!T)
    return {};
  if (auto *ET = T->getAs<EnumType>()) {
    auto Result =
        ET->getDecl()->lookup(NameFactory(ET->getDecl()->getASTContext()));
    return {Result.begin(), Result.end()};
  }
  if (auto *RD = resolveTypeToRecordDecl(T)) {
    if (!RD->hasDefinition())
      return {};
    RD = RD->getDefinition();
    DeclarationName Name = NameFactory(RD->getASTContext());
    return RD->lookupDependentName(Name, Filter);
  }
  return {};
}

const auto NonStaticFilter = [](const NamedDecl *D) {
  return D->isCXXInstanceMember();
};
const auto StaticFilter = [](const NamedDecl *D) {
  return !D->isCXXInstanceMember();
};
const auto ValueFilter = [](const NamedDecl *D) { return isa<ValueDecl>(D); };

// Given the type T of a dependent expression that appears of the LHS of a
// "->", heuristically find a corresponding pointee type in whose scope we
// could look up the name appearing on the RHS.
const Type *getPointeeType(const Type *T) {
  if (!T)
    return nullptr;

  if (T->isPointerType()) {
    return T->getAs<PointerType>()->getPointeeType().getTypePtrOrNull();
  }

  // Try to handle smart pointer types.

  // Look up operator-> in the primary template. If we find one, it's probably a
  // smart pointer type.
  auto ArrowOps = getMembersReferencedViaDependentName(
      T,
      [](ASTContext &Ctx) {
        return Ctx.DeclarationNames.getCXXOperatorName(OO_Arrow);
      },
      NonStaticFilter);
  if (ArrowOps.empty())
    return nullptr;

  // Getting the return type of the found operator-> method decl isn't useful,
  // because we discarded template arguments to perform lookup in the primary
  // template scope, so the return type would just have the form U* where U is a
  // template parameter type.
  // Instead, just handle the common case where the smart pointer type has the
  // form of SmartPtr<X, ...>, and assume X is the pointee type.
  auto *TST = T->getAs<TemplateSpecializationType>();
  if (!TST)
    return nullptr;
  if (TST->getNumArgs() == 0)
    return nullptr;
  const TemplateArgument &FirstArg = TST->getArg(0);
  if (FirstArg.getKind() != TemplateArgument::Type)
    return nullptr;
  return FirstArg.getAsType().getTypePtrOrNull();
}

// Forward declaration, needed as this function is mutually recursive
// with resolveExprToDecls.
const Type *resolveExprToType(const Expr *E);

// Try to heuristically resolve a possibly-dependent expression `E` to one
// or more declarations that it likely references.
std::vector<const NamedDecl *> resolveExprToDecls(const Expr *E) {
  if (const auto *ME = dyn_cast<CXXDependentScopeMemberExpr>(E)) {
    const Type *BaseType = ME->getBaseType().getTypePtrOrNull();
    if (ME->isArrow()) {
      BaseType = getPointeeType(BaseType);
    }
    if (!BaseType)
      return {};
    if (const auto *BT = BaseType->getAs<BuiltinType>()) {
      // If BaseType is the type of a dependent expression, it's just
      // represented as BultinType::Dependent which gives us no information. We
      // can get further by analyzing the depedent expression.
      Expr *Base = ME->isImplicitAccess() ? nullptr : ME->getBase();
      if (Base && BT->getKind() == BuiltinType::Dependent) {
        BaseType = resolveExprToType(Base);
      }
    }
    return getMembersReferencedViaDependentName(
        BaseType, [ME](ASTContext &) { return ME->getMember(); },
        NonStaticFilter);
  }
  if (const auto *RE = dyn_cast<DependentScopeDeclRefExpr>(E)) {
    return getMembersReferencedViaDependentName(
        RE->getQualifier()->getAsType(),
        [RE](ASTContext &) { return RE->getDeclName(); }, StaticFilter);
  }
  if (const auto *CE = dyn_cast<CallExpr>(E)) {
    const auto *CalleeType = resolveExprToType(CE->getCallee());
    if (!CalleeType)
      return {};
    if (const auto *FnTypePtr = CalleeType->getAs<PointerType>())
      CalleeType = FnTypePtr->getPointeeType().getTypePtr();
    if (const FunctionType *FnType = CalleeType->getAs<FunctionType>()) {
      if (const auto *D =
              resolveTypeToRecordDecl(FnType->getReturnType().getTypePtr())) {
        return {D};
      }
    }
  }
  if (const auto *ME = dyn_cast<MemberExpr>(E))
    return {ME->getMemberDecl()};
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return {DRE->getFoundDecl()};
  return {};
}

// Try to heuristically resolve the type of a possibly-dependent expression `E`.
const Type *resolveExprToType(const Expr *E) {
  std::vector<const NamedDecl *> Decls = resolveExprToDecls(E);
  if (Decls.size() != 1) // Names an overload set -- just bail.
    return nullptr;
  if (const auto *TD = dyn_cast<TypeDecl>(Decls[0])) {
    return TD->getTypeForDecl();
  } else if (const auto *VD = dyn_cast<ValueDecl>(Decls[0])) {
    return VD->getType().getTypePtrOrNull();
  }
  return nullptr;
}

const NamedDecl *getTemplatePattern(const NamedDecl *D) {
  if (const CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(D)) {
    if (const auto *Result = CRD->getTemplateInstantiationPattern())
      return Result;
    // getTemplateInstantiationPattern returns null if the Specialization is
    // incomplete (e.g. the type didn't need to be complete), fall back to the
    // primary template.
    if (CRD->getTemplateSpecializationKind() == TSK_Undeclared)
      if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(CRD))
        return Spec->getSpecializedTemplate()->getTemplatedDecl();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    return FD->getTemplateInstantiationPattern();
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    // Hmm: getTIP returns its arg if it's not an instantiation?!
    VarDecl *T = VD->getTemplateInstantiationPattern();
    return (T == D) ? nullptr : T;
  } else if (const auto *ED = dyn_cast<EnumDecl>(D)) {
    return ED->getInstantiatedFromMemberEnum();
  } else if (isa<FieldDecl>(D) || isa<TypedefNameDecl>(D)) {
    if (const auto *Parent = llvm::dyn_cast<NamedDecl>(D->getDeclContext()))
      if (const DeclContext *ParentPat =
              dyn_cast_or_null<DeclContext>(getTemplatePattern(Parent)))
        for (const NamedDecl *BaseND : ParentPat->lookup(D->getDeclName()))
          if (!BaseND->isImplicit() && BaseND->getKind() == D->getKind())
            return BaseND;
  } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    if (const auto *ED = dyn_cast<EnumDecl>(ECD->getDeclContext())) {
      if (const EnumDecl *Pattern = ED->getInstantiatedFromMemberEnum()) {
        for (const NamedDecl *BaseECD : Pattern->lookup(ECD->getDeclName()))
          return BaseECD;
      }
    }
  }
  return nullptr;
}

// TargetFinder locates the entities that an AST node refers to.
//
// Typically this is (possibly) one declaration and (possibly) one type, but
// may be more:
//  - for ambiguous nodes like OverloadExpr
//  - if we want to include e.g. both typedefs and the underlying type
//
// This is organized as a set of mutually recursive helpers for particular node
// types, but for most nodes this is a short walk rather than a deep traversal.
//
// It's tempting to do e.g. typedef resolution as a second normalization step,
// after finding the 'primary' decl etc. But we do this monolithically instead
// because:
//  - normalization may require these traversals again (e.g. unwrapping a
//    typedef reveals a decltype which must be traversed)
//  - it doesn't simplify that much, e.g. the first stage must still be able
//    to yield multiple decls to handle OverloadExpr
//  - there are cases where it's required for correctness. e.g:
//      template<class X> using pvec = vector<x*>; pvec<int> x;
//    There's no Decl `pvec<int>`, we must choose `pvec<X>` or `vector<int*>`
//    and both are lossy. We must know upfront what the caller ultimately wants.
//
// FIXME: improve common dependent scope using name lookup in primary templates.
// We currently handle DependentScopeDeclRefExpr and
// CXXDependentScopeMemberExpr, but some other constructs remain to be handled:
//  - DependentTemplateSpecializationType,
//  - DependentNameType
//  - UnresolvedUsingTypenameDecl
struct TargetFinder {
  using RelSet = DeclRelationSet;
  using Rel = DeclRelation;

private:
  llvm::SmallDenseMap<const NamedDecl *,
                      std::pair<RelSet, /*InsertionOrder*/ size_t>>
      Decls;
  RelSet Flags;

  template <typename T> void debug(T &Node, RelSet Flags) {
    dlog("visit [{0}] {1}", Flags,
         nodeToString(ast_type_traits::DynTypedNode::create(Node)));
  }

  void report(const NamedDecl *D, RelSet Flags) {
    dlog("--> [{0}] {1}", Flags,
         nodeToString(ast_type_traits::DynTypedNode::create(*D)));
    auto It = Decls.try_emplace(D, std::make_pair(Flags, Decls.size()));
    // If already exists, update the flags.
    if (!It.second)
      It.first->second.first |= Flags;
  }

public:
  llvm::SmallVector<std::pair<const NamedDecl *, RelSet>, 1> takeDecls() const {
    using ValTy = std::pair<const NamedDecl *, RelSet>;
    llvm::SmallVector<ValTy, 1> Result;
    Result.resize(Decls.size());
    for (const auto &Elem : Decls)
      Result[Elem.second.second] = {Elem.first, Elem.second.first};
    return Result;
  }

  void add(const Decl *Dcl, RelSet Flags) {
    const NamedDecl *D = llvm::dyn_cast_or_null<NamedDecl>(Dcl);
    if (!D)
      return;
    debug(*D, Flags);
    if (const UsingDirectiveDecl *UDD = llvm::dyn_cast<UsingDirectiveDecl>(D))
      D = UDD->getNominatedNamespaceAsWritten();

    if (const TypedefNameDecl *TND = dyn_cast<TypedefNameDecl>(D)) {
      add(TND->getUnderlyingType(), Flags | Rel::Underlying);
      Flags |= Rel::Alias; // continue with the alias.
    } else if (const UsingDecl *UD = dyn_cast<UsingDecl>(D)) {
      // no Underlying as this is a non-renaming alias.
      for (const UsingShadowDecl *S : UD->shadows())
        add(S->getUnderlyingDecl(), Flags);
      Flags |= Rel::Alias; // continue with the alias.
    } else if (const auto *NAD = dyn_cast<NamespaceAliasDecl>(D)) {
      add(NAD->getUnderlyingDecl(), Flags | Rel::Underlying);
      Flags |= Rel::Alias; // continue with the alias
    } else if (const UnresolvedUsingValueDecl *UUVD =
                   dyn_cast<UnresolvedUsingValueDecl>(D)) {
      for (const NamedDecl *Target : getMembersReferencedViaDependentName(
               UUVD->getQualifier()->getAsType(),
               [UUVD](ASTContext &) { return UUVD->getNameInfo().getName(); },
               ValueFilter)) {
        add(Target, Flags); // no Underlying as this is a non-renaming alias
      }
      Flags |= Rel::Alias; // continue with the alias
    } else if (const UsingShadowDecl *USD = dyn_cast<UsingShadowDecl>(D)) {
      // Include the using decl, but don't traverse it. This may end up
      // including *all* shadows, which we don't want.
      report(USD->getUsingDecl(), Flags | Rel::Alias);
      // Shadow decls are synthetic and not themselves interesting.
      // Record the underlying decl instead, if allowed.
      D = USD->getTargetDecl();
    } else if (const auto *DG = dyn_cast<CXXDeductionGuideDecl>(D)) {
      D = DG->getDeducedTemplate();
    } else if (const ObjCImplementationDecl *IID =
                   dyn_cast<ObjCImplementationDecl>(D)) {
      // Treat ObjC{Interface,Implementation}Decl as if they were a decl/def
      // pair as long as the interface isn't implicit.
      if (const auto *CID = IID->getClassInterface())
        if (const auto *DD = CID->getDefinition())
          if (!DD->isImplicitInterfaceDecl())
            D = DD;
    } else if (const ObjCCategoryImplDecl *CID =
                   dyn_cast<ObjCCategoryImplDecl>(D)) {
      // Treat ObjC{Category,CategoryImpl}Decl as if they were a decl/def pair.
      D = CID->getCategoryDecl();
    }
    if (!D)
      return;

    if (const Decl *Pat = getTemplatePattern(D)) {
      assert(Pat != D);
      add(Pat, Flags | Rel::TemplatePattern);
      // Now continue with the instantiation.
      Flags |= Rel::TemplateInstantiation;
    }

    report(D, Flags);
  }

  void add(const Stmt *S, RelSet Flags) {
    if (!S)
      return;
    debug(*S, Flags);
    struct Visitor : public ConstStmtVisitor<Visitor> {
      TargetFinder &Outer;
      RelSet Flags;
      Visitor(TargetFinder &Outer, RelSet Flags) : Outer(Outer), Flags(Flags) {}

      void VisitCallExpr(const CallExpr *CE) {
        Outer.add(CE->getCalleeDecl(), Flags);
      }
      void VisitConceptSpecializationExpr(const ConceptSpecializationExpr *E) {
        Outer.add(E->getNamedConcept(), Flags);
      }
      void VisitDeclRefExpr(const DeclRefExpr *DRE) {
        const Decl *D = DRE->getDecl();
        // UsingShadowDecl allows us to record the UsingDecl.
        // getFoundDecl() returns the wrong thing in other cases (templates).
        if (auto *USD = llvm::dyn_cast<UsingShadowDecl>(DRE->getFoundDecl()))
          D = USD;
        Outer.add(D, Flags);
      }
      void VisitMemberExpr(const MemberExpr *ME) {
        const Decl *D = ME->getMemberDecl();
        if (auto *USD =
                llvm::dyn_cast<UsingShadowDecl>(ME->getFoundDecl().getDecl()))
          D = USD;
        Outer.add(D, Flags);
      }
      void VisitOverloadExpr(const OverloadExpr *OE) {
        for (auto *D : OE->decls())
          Outer.add(D, Flags);
      }
      void VisitSizeOfPackExpr(const SizeOfPackExpr *SE) {
        Outer.add(SE->getPack(), Flags);
      }
      void VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
        Outer.add(CCE->getConstructor(), Flags);
      }
      void VisitDesignatedInitExpr(const DesignatedInitExpr *DIE) {
        for (const DesignatedInitExpr::Designator &D :
             llvm::reverse(DIE->designators()))
          if (D.isFieldDesignator()) {
            Outer.add(D.getField(), Flags);
            // We don't know which designator was intended, we assume the outer.
            break;
          }
      }
      void VisitGotoStmt(const GotoStmt *Goto) {
        if (auto *LabelDecl = Goto->getLabel())
          Outer.add(LabelDecl, Flags);
      }
      void VisitLabelStmt(const LabelStmt *Label) {
        if (auto *LabelDecl = Label->getDecl())
          Outer.add(LabelDecl, Flags);
      }
      void
      VisitCXXDependentScopeMemberExpr(const CXXDependentScopeMemberExpr *E) {
        for (const NamedDecl *D : resolveExprToDecls(E)) {
          Outer.add(D, Flags);
        }
      }
      void VisitDependentScopeDeclRefExpr(const DependentScopeDeclRefExpr *E) {
        for (const NamedDecl *D : resolveExprToDecls(E)) {
          Outer.add(D, Flags);
        }
      }
      void VisitObjCIvarRefExpr(const ObjCIvarRefExpr *OIRE) {
        Outer.add(OIRE->getDecl(), Flags);
      }
      void VisitObjCMessageExpr(const ObjCMessageExpr *OME) {
        Outer.add(OME->getMethodDecl(), Flags);
      }
      void VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *OPRE) {
        if (OPRE->isExplicitProperty())
          Outer.add(OPRE->getExplicitProperty(), Flags);
        else {
          if (OPRE->isMessagingGetter())
            Outer.add(OPRE->getImplicitPropertyGetter(), Flags);
          if (OPRE->isMessagingSetter())
            Outer.add(OPRE->getImplicitPropertySetter(), Flags);
        }
      }
      void VisitObjCProtocolExpr(const ObjCProtocolExpr *OPE) {
        Outer.add(OPE->getProtocol(), Flags);
      }
      void VisitOpaqueValueExpr(const OpaqueValueExpr *OVE) {
        Outer.add(OVE->getSourceExpr(), Flags);
      }
      void VisitPseudoObjectExpr(const PseudoObjectExpr *POE) {
        Outer.add(POE->getSyntacticForm(), Flags);
      }
      void VisitCXXNewExpr(const CXXNewExpr *CNE) {
        Outer.add(CNE->getOperatorNew(), Flags);
      }
      void VisitCXXDeleteExpr(const CXXDeleteExpr *CDE) {
        Outer.add(CDE->getOperatorDelete(), Flags);
      }
    };
    Visitor(*this, Flags).Visit(S);
  }

  void add(QualType T, RelSet Flags) {
    if (T.isNull())
      return;
    debug(T, Flags);
    struct Visitor : public TypeVisitor<Visitor> {
      TargetFinder &Outer;
      RelSet Flags;
      Visitor(TargetFinder &Outer, RelSet Flags) : Outer(Outer), Flags(Flags) {}

      void VisitTagType(const TagType *TT) {
        Outer.add(TT->getAsTagDecl(), Flags);
      }

      void VisitElaboratedType(const ElaboratedType *ET) {
        Outer.add(ET->desugar(), Flags);
      }

      void VisitInjectedClassNameType(const InjectedClassNameType *ICNT) {
        Outer.add(ICNT->getDecl(), Flags);
      }

      void VisitDecltypeType(const DecltypeType *DTT) {
        Outer.add(DTT->getUnderlyingType(), Flags | Rel::Underlying);
      }
      void VisitDeducedType(const DeducedType *DT) {
        // FIXME: In practice this doesn't work: the AutoType you find inside
        // TypeLoc never has a deduced type. https://llvm.org/PR42914
        Outer.add(DT->getDeducedType(), Flags | Rel::Underlying);
      }
      void VisitDeducedTemplateSpecializationType(
          const DeducedTemplateSpecializationType *DTST) {
        // FIXME: This is a workaround for https://llvm.org/PR42914,
        // which is causing DTST->getDeducedType() to be empty. We
        // fall back to the template pattern and miss the instantiation
        // even when it's known in principle. Once that bug is fixed,
        // this method can be removed (the existing handling in
        // VisitDeducedType() is sufficient).
        if (auto *TD = DTST->getTemplateName().getAsTemplateDecl())
          Outer.add(TD->getTemplatedDecl(), Flags | Rel::TemplatePattern);
      }
      void VisitTypedefType(const TypedefType *TT) {
        Outer.add(TT->getDecl(), Flags);
      }
      void
      VisitTemplateSpecializationType(const TemplateSpecializationType *TST) {
        // Have to handle these case-by-case.

        // templated type aliases: there's no specialized/instantiated using
        // decl to point to. So try to find a decl for the underlying type
        // (after substitution), and failing that point to the (templated) using
        // decl.
        if (TST->isTypeAlias()) {
          Outer.add(TST->getAliasedType(), Flags | Rel::Underlying);
          // Don't *traverse* the alias, which would result in traversing the
          // template of the underlying type.
          Outer.report(
              TST->getTemplateName().getAsTemplateDecl()->getTemplatedDecl(),
              Flags | Rel::Alias | Rel::TemplatePattern);
        }
        // specializations of template template parameters aren't instantiated
        // into decls, so they must refer to the parameter itself.
        else if (const auto *Parm =
                     llvm::dyn_cast_or_null<TemplateTemplateParmDecl>(
                         TST->getTemplateName().getAsTemplateDecl()))
          Outer.add(Parm, Flags);
        // class template specializations have a (specialized) CXXRecordDecl.
        else if (const CXXRecordDecl *RD = TST->getAsCXXRecordDecl())
          Outer.add(RD, Flags); // add(Decl) will despecialize if needed.
        else {
          // fallback: the (un-specialized) declaration from primary template.
          if (auto *TD = TST->getTemplateName().getAsTemplateDecl())
            Outer.add(TD->getTemplatedDecl(), Flags | Rel::TemplatePattern);
        }
      }
      void VisitTemplateTypeParmType(const TemplateTypeParmType *TTPT) {
        Outer.add(TTPT->getDecl(), Flags);
      }
      void VisitObjCInterfaceType(const ObjCInterfaceType *OIT) {
        Outer.add(OIT->getDecl(), Flags);
      }
      void VisitObjCObjectType(const ObjCObjectType *OOT) {
        // FIXME: ObjCObjectTypeLoc has no children for the protocol list, so
        // there is no node in id<Foo> that refers to ObjCProtocolDecl Foo.
        if (OOT->isObjCQualifiedId() && OOT->getNumProtocols() == 1)
          Outer.add(OOT->getProtocol(0), Flags);
      }
    };
    Visitor(*this, Flags).Visit(T.getTypePtr());
  }

  void add(const NestedNameSpecifier *NNS, RelSet Flags) {
    if (!NNS)
      return;
    debug(*NNS, Flags);
    switch (NNS->getKind()) {
    case NestedNameSpecifier::Identifier:
      return;
    case NestedNameSpecifier::Namespace:
      add(NNS->getAsNamespace(), Flags);
      return;
    case NestedNameSpecifier::NamespaceAlias:
      add(NNS->getAsNamespaceAlias(), Flags);
      return;
    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      add(QualType(NNS->getAsType(), 0), Flags);
      return;
    case NestedNameSpecifier::Global:
      // This should be TUDecl, but we can't get a pointer to it!
      return;
    case NestedNameSpecifier::Super:
      add(NNS->getAsRecordDecl(), Flags);
      return;
    }
    llvm_unreachable("unhandled NestedNameSpecifier::SpecifierKind");
  }

  void add(const CXXCtorInitializer *CCI, RelSet Flags) {
    if (!CCI)
      return;
    debug(*CCI, Flags);

    if (CCI->isAnyMemberInitializer())
      add(CCI->getAnyMember(), Flags);
    // Constructor calls contain a TypeLoc node, so we don't handle them here.
  }

  void add(const TemplateArgument &Arg, RelSet Flags) {
    // Only used for template template arguments.
    // For type and non-type template arguments, SelectionTree
    // will hit a more specific node (e.g. a TypeLoc or a
    // DeclRefExpr).
    if (Arg.getKind() == TemplateArgument::Template ||
        Arg.getKind() == TemplateArgument::TemplateExpansion) {
      if (TemplateDecl *TD = Arg.getAsTemplate().getAsTemplateDecl()) {
        report(TD, Flags);
      }
    }
  }
};

} // namespace

llvm::SmallVector<std::pair<const NamedDecl *, DeclRelationSet>, 1>
allTargetDecls(const ast_type_traits::DynTypedNode &N) {
  dlog("allTargetDecls({0})", nodeToString(N));
  TargetFinder Finder;
  DeclRelationSet Flags;
  if (const Decl *D = N.get<Decl>())
    Finder.add(D, Flags);
  else if (const Stmt *S = N.get<Stmt>())
    Finder.add(S, Flags);
  else if (const NestedNameSpecifierLoc *NNSL = N.get<NestedNameSpecifierLoc>())
    Finder.add(NNSL->getNestedNameSpecifier(), Flags);
  else if (const NestedNameSpecifier *NNS = N.get<NestedNameSpecifier>())
    Finder.add(NNS, Flags);
  else if (const TypeLoc *TL = N.get<TypeLoc>())
    Finder.add(TL->getType(), Flags);
  else if (const QualType *QT = N.get<QualType>())
    Finder.add(*QT, Flags);
  else if (const CXXCtorInitializer *CCI = N.get<CXXCtorInitializer>())
    Finder.add(CCI, Flags);
  else if (const TemplateArgumentLoc *TAL = N.get<TemplateArgumentLoc>())
    Finder.add(TAL->getArgument(), Flags);

  return Finder.takeDecls();
}

llvm::SmallVector<const NamedDecl *, 1>
targetDecl(const ast_type_traits::DynTypedNode &N, DeclRelationSet Mask) {
  llvm::SmallVector<const NamedDecl *, 1> Result;
  for (const auto &Entry : allTargetDecls(N)) {
    if (!(Entry.second & ~Mask))
      Result.push_back(Entry.first);
  }
  return Result;
}

llvm::SmallVector<const NamedDecl *, 1>
explicitReferenceTargets(DynTypedNode N, DeclRelationSet Mask) {
  assert(!(Mask & (DeclRelation::TemplatePattern |
                   DeclRelation::TemplateInstantiation)) &&
         "explicitReferenceTargets handles templates on its own");
  auto Decls = allTargetDecls(N);

  // We prefer to return template instantiation, but fallback to template
  // pattern if instantiation is not available.
  Mask |= DeclRelation::TemplatePattern | DeclRelation::TemplateInstantiation;

  llvm::SmallVector<const NamedDecl *, 1> TemplatePatterns;
  llvm::SmallVector<const NamedDecl *, 1> Targets;
  bool SeenTemplateInstantiations = false;
  for (auto &D : Decls) {
    if (D.second & ~Mask)
      continue;
    if (D.second & DeclRelation::TemplatePattern) {
      TemplatePatterns.push_back(D.first);
      continue;
    }
    if (D.second & DeclRelation::TemplateInstantiation)
      SeenTemplateInstantiations = true;
    Targets.push_back(D.first);
  }
  if (!SeenTemplateInstantiations)
    Targets.insert(Targets.end(), TemplatePatterns.begin(),
                   TemplatePatterns.end());
  return Targets;
}

namespace {
llvm::SmallVector<ReferenceLoc, 2> refInDecl(const Decl *D) {
  struct Visitor : ConstDeclVisitor<Visitor> {
    llvm::SmallVector<ReferenceLoc, 2> Refs;

    void VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
      // We want to keep it as non-declaration references, as the
      // "using namespace" declaration doesn't have a name.
      Refs.push_back(ReferenceLoc{D->getQualifierLoc(),
                                  D->getIdentLocation(),
                                  /*IsDecl=*/false,
                                  {D->getNominatedNamespaceAsWritten()}});
    }

    void VisitUsingDecl(const UsingDecl *D) {
      // "using ns::identifier;" is a non-declaration reference.
      Refs.push_back(
          ReferenceLoc{D->getQualifierLoc(), D->getLocation(), /*IsDecl=*/false,
                       explicitReferenceTargets(DynTypedNode::create(*D),
                                                DeclRelation::Underlying)});
    }

    void VisitNamespaceAliasDecl(const NamespaceAliasDecl *D) {
      // For namespace alias, "namespace Foo = Target;", we add two references.
      // Add a declaration reference for Foo.
      VisitNamedDecl(D);
      // Add a non-declaration reference for Target.
      Refs.push_back(ReferenceLoc{D->getQualifierLoc(),
                                  D->getTargetNameLoc(),
                                  /*IsDecl=*/false,
                                  {D->getAliasedNamespace()}});
    }

    void VisitNamedDecl(const NamedDecl *ND) {
      // We choose to ignore {Class, Function, Var, TypeAlias}TemplateDecls. As
      // as their underlying decls, covering the same range, will be visited.
      if (llvm::isa<ClassTemplateDecl>(ND) ||
          llvm::isa<FunctionTemplateDecl>(ND) ||
          llvm::isa<VarTemplateDecl>(ND) ||
          llvm::isa<TypeAliasTemplateDecl>(ND))
        return;
      // FIXME: decide on how to surface destructors when we need them.
      if (llvm::isa<CXXDestructorDecl>(ND))
        return;
      // Filter anonymous decls, name location will point outside the name token
      // and the clients are not prepared to handle that.
      if (ND->getDeclName().isIdentifier() &&
          !ND->getDeclName().getAsIdentifierInfo())
        return;
      Refs.push_back(ReferenceLoc{getQualifierLoc(*ND),
                                  ND->getLocation(),
                                  /*IsDecl=*/true,
                                  {ND}});
    }

    void VisitCXXDeductionGuideDecl(const CXXDeductionGuideDecl *DG) {
      // The class template name in a deduction guide targets the class
      // template.
      Refs.push_back(ReferenceLoc{DG->getQualifierLoc(),
                                  DG->getNameInfo().getLoc(),
                                  /*IsDecl=*/false,
                                  {DG->getDeducedTemplate()}});
    }
  };

  Visitor V;
  V.Visit(D);
  return V.Refs;
}

llvm::SmallVector<ReferenceLoc, 2> refInStmt(const Stmt *S) {
  struct Visitor : ConstStmtVisitor<Visitor> {
    // FIXME: handle more complicated cases: more ObjC, designated initializers.
    llvm::SmallVector<ReferenceLoc, 2> Refs;

    void VisitConceptSpecializationExpr(const ConceptSpecializationExpr *E) {
      Refs.push_back(ReferenceLoc{E->getNestedNameSpecifierLoc(),
                                  E->getConceptNameLoc(),
                                  /*IsDecl=*/false,
                                  {E->getNamedConcept()}});
    }

    void VisitDeclRefExpr(const DeclRefExpr *E) {
      Refs.push_back(ReferenceLoc{E->getQualifierLoc(),
                                  E->getNameInfo().getLoc(),
                                  /*IsDecl=*/false,
                                  {E->getFoundDecl()}});
    }

    void VisitDependentScopeDeclRefExpr(const DependentScopeDeclRefExpr *E) {
      Refs.push_back(ReferenceLoc{
          E->getQualifierLoc(), E->getNameInfo().getLoc(), /*IsDecl=*/false,
          explicitReferenceTargets(DynTypedNode::create(*E), {})});
    }

    void VisitMemberExpr(const MemberExpr *E) {
      // Skip destructor calls to avoid duplication: TypeLoc within will be
      // visited separately.
      if (llvm::dyn_cast<CXXDestructorDecl>(E->getFoundDecl().getDecl()))
        return;
      Refs.push_back(ReferenceLoc{E->getQualifierLoc(),
                                  E->getMemberNameInfo().getLoc(),
                                  /*IsDecl=*/false,
                                  {E->getFoundDecl()}});
    }

    void
    VisitCXXDependentScopeMemberExpr(const CXXDependentScopeMemberExpr *E) {
      Refs.push_back(
          ReferenceLoc{E->getQualifierLoc(), E->getMemberNameInfo().getLoc(),
                       /*IsDecl=*/false,
                       explicitReferenceTargets(DynTypedNode::create(*E), {})});
    }

    void VisitOverloadExpr(const OverloadExpr *E) {
      Refs.push_back(ReferenceLoc{E->getQualifierLoc(),
                                  E->getNameInfo().getLoc(),
                                  /*IsDecl=*/false,
                                  llvm::SmallVector<const NamedDecl *, 1>(
                                      E->decls().begin(), E->decls().end())});
    }

    void VisitSizeOfPackExpr(const SizeOfPackExpr *E) {
      Refs.push_back(ReferenceLoc{NestedNameSpecifierLoc(),
                                  E->getPackLoc(),
                                  /*IsDecl=*/false,
                                  {E->getPack()}});
    }

    void VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *E) {
      Refs.push_back(ReferenceLoc{
          NestedNameSpecifierLoc(), E->getLocation(),
          /*IsDecl=*/false,
          // Select the getter, setter, or @property depending on the call.
          explicitReferenceTargets(DynTypedNode::create(*E), {})});
    }

    void VisitDesignatedInitExpr(const DesignatedInitExpr *DIE) {
      for (const DesignatedInitExpr::Designator &D : DIE->designators()) {
        if (!D.isFieldDesignator())
          continue;

        llvm::SmallVector<const NamedDecl *, 1> Targets;
        if (D.getField())
          Targets.push_back(D.getField());
        Refs.push_back(ReferenceLoc{NestedNameSpecifierLoc(), D.getFieldLoc(),
                                    /*IsDecl=*/false, std::move(Targets)});
      }
    }

    void VisitGotoStmt(const GotoStmt *GS) {
      llvm::SmallVector<const NamedDecl *, 1> Targets;
      if (const auto *L = GS->getLabel())
        Targets.push_back(L);
      Refs.push_back(ReferenceLoc{NestedNameSpecifierLoc(), GS->getLabelLoc(),
                                  /*IsDecl=*/false, std::move(Targets)});
    }

    void VisitLabelStmt(const LabelStmt *LS) {
      Refs.push_back(ReferenceLoc{NestedNameSpecifierLoc(),
                                  LS->getIdentLoc(),
                                  /*IsDecl=*/true,
                                  {LS->getDecl()}});
    }
  };

  Visitor V;
  V.Visit(S);
  return V.Refs;
}

llvm::SmallVector<ReferenceLoc, 2> refInTypeLoc(TypeLoc L) {
  struct Visitor : TypeLocVisitor<Visitor> {
    llvm::Optional<ReferenceLoc> Ref;

    void VisitElaboratedTypeLoc(ElaboratedTypeLoc L) {
      // We only know about qualifier, rest if filled by inner locations.
      Visit(L.getNamedTypeLoc().getUnqualifiedLoc());
      // Fill in the qualifier.
      if (!Ref)
        return;
      assert(!Ref->Qualifier.hasQualifier() && "qualifier already set");
      Ref->Qualifier = L.getQualifierLoc();
    }

    void VisitTagTypeLoc(TagTypeLoc L) {
      Ref = ReferenceLoc{NestedNameSpecifierLoc(),
                         L.getNameLoc(),
                         /*IsDecl=*/false,
                         {L.getDecl()}};
    }

    void VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc L) {
      Ref = ReferenceLoc{NestedNameSpecifierLoc(),
                         L.getNameLoc(),
                         /*IsDecl=*/false,
                         {L.getDecl()}};
    }

    void VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc L) {
      // We must ensure template type aliases are included in results if they
      // were written in the source code, e.g. in
      //    template <class T> using valias = vector<T>;
      //    ^valias<int> x;
      // 'explicitReferenceTargets' will return:
      //    1. valias with mask 'Alias'.
      //    2. 'vector<int>' with mask 'Underlying'.
      //  we want to return only #1 in this case.
      Ref = ReferenceLoc{
          NestedNameSpecifierLoc(), L.getTemplateNameLoc(), /*IsDecl=*/false,
          explicitReferenceTargets(DynTypedNode::create(L.getType()),
                                   DeclRelation::Alias)};
    }
    void VisitDeducedTemplateSpecializationTypeLoc(
        DeducedTemplateSpecializationTypeLoc L) {
      Ref = ReferenceLoc{
          NestedNameSpecifierLoc(), L.getNameLoc(), /*IsDecl=*/false,
          explicitReferenceTargets(DynTypedNode::create(L.getType()),
                                   DeclRelation::Alias)};
    }

    void VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
      Ref = ReferenceLoc{NestedNameSpecifierLoc(),
                         TL.getNameLoc(),
                         /*IsDecl=*/false,
                         {TL.getDecl()}};
    }

    void VisitDependentTemplateSpecializationTypeLoc(
        DependentTemplateSpecializationTypeLoc L) {
      Ref = ReferenceLoc{
          L.getQualifierLoc(), L.getTemplateNameLoc(), /*IsDecl=*/false,
          explicitReferenceTargets(DynTypedNode::create(L.getType()), {})};
    }

    void VisitDependentNameTypeLoc(DependentNameTypeLoc L) {
      Ref = ReferenceLoc{
          L.getQualifierLoc(), L.getNameLoc(), /*IsDecl=*/false,
          explicitReferenceTargets(DynTypedNode::create(L.getType()), {})};
    }

    void VisitTypedefTypeLoc(TypedefTypeLoc L) {
      Ref = ReferenceLoc{NestedNameSpecifierLoc(),
                         L.getNameLoc(),
                         /*IsDecl=*/false,
                         {L.getTypedefNameDecl()}};
    }
  };

  Visitor V;
  V.Visit(L.getUnqualifiedLoc());
  if (!V.Ref)
    return {};
  return {*V.Ref};
}

class ExplicitReferenceCollector
    : public RecursiveASTVisitor<ExplicitReferenceCollector> {
public:
  ExplicitReferenceCollector(llvm::function_ref<void(ReferenceLoc)> Out)
      : Out(Out) {
    assert(Out);
  }

  bool VisitTypeLoc(TypeLoc TTL) {
    if (TypeLocsToSkip.count(TTL.getBeginLoc().getRawEncoding()))
      return true;
    visitNode(DynTypedNode::create(TTL));
    return true;
  }

  bool TraverseElaboratedTypeLoc(ElaboratedTypeLoc L) {
    // ElaboratedTypeLoc will reports information for its inner type loc.
    // Otherwise we loose information about inner types loc's qualifier.
    TypeLoc Inner = L.getNamedTypeLoc().getUnqualifiedLoc();
    TypeLocsToSkip.insert(Inner.getBeginLoc().getRawEncoding());
    return RecursiveASTVisitor::TraverseElaboratedTypeLoc(L);
  }

  bool VisitStmt(Stmt *S) {
    visitNode(DynTypedNode::create(*S));
    return true;
  }

  bool TraverseOpaqueValueExpr(OpaqueValueExpr *OVE) {
    visitNode(DynTypedNode::create(*OVE));
    // Not clear why the source expression is skipped by default...
    // FIXME: can we just make RecursiveASTVisitor do this?
    return RecursiveASTVisitor::TraverseStmt(OVE->getSourceExpr());
  }

  bool TraversePseudoObjectExpr(PseudoObjectExpr *POE) {
    visitNode(DynTypedNode::create(*POE));
    // Traverse only the syntactic form to find the *written* references.
    // (The semantic form also contains lots of duplication)
    return RecursiveASTVisitor::TraverseStmt(POE->getSyntacticForm());
  }

  // We re-define Traverse*, since there's no corresponding Visit*.
  // TemplateArgumentLoc is the only way to get locations for references to
  // template template parameters.
  bool TraverseTemplateArgumentLoc(TemplateArgumentLoc A) {
    llvm::SmallVector<const NamedDecl *, 1> Targets;
    switch (A.getArgument().getKind()) {
    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
      if (const auto *D = A.getArgument()
                              .getAsTemplateOrTemplatePattern()
                              .getAsTemplateDecl())
        Targets.push_back(D);
      reportReference(ReferenceLoc{A.getTemplateQualifierLoc(),
                                   A.getTemplateNameLoc(),
                                   /*IsDecl=*/false, Targets},
                      DynTypedNode::create(A.getArgument()));
      break;
    case TemplateArgument::Declaration:
      break; // FIXME: can this actually happen in TemplateArgumentLoc?
    case TemplateArgument::Integral:
    case TemplateArgument::Null:
    case TemplateArgument::NullPtr:
      break; // no references.
    case TemplateArgument::Pack:
    case TemplateArgument::Type:
    case TemplateArgument::Expression:
      break; // Handled by VisitType and VisitExpression.
    };
    return RecursiveASTVisitor::TraverseTemplateArgumentLoc(A);
  }

  bool VisitDecl(Decl *D) {
    visitNode(DynTypedNode::create(*D));
    return true;
  }

  // We have to use Traverse* because there is no corresponding Visit*.
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc L) {
    if (!L.getNestedNameSpecifier())
      return true;
    visitNode(DynTypedNode::create(L));
    // Inner type is missing information about its qualifier, skip it.
    if (auto TL = L.getTypeLoc())
      TypeLocsToSkip.insert(TL.getBeginLoc().getRawEncoding());
    return RecursiveASTVisitor::TraverseNestedNameSpecifierLoc(L);
  }

  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) {
    visitNode(DynTypedNode::create(*Init));
    return RecursiveASTVisitor::TraverseConstructorInitializer(Init);
  }

private:
  /// Obtain information about a reference directly defined in \p N. Does not
  /// recurse into child nodes, e.g. do not expect references for constructor
  /// initializers
  ///
  /// Any of the fields in the returned structure can be empty, but not all of
  /// them, e.g.
  ///   - for implicitly generated nodes (e.g. MemberExpr from range-based-for),
  ///     source location information may be missing,
  ///   - for dependent code, targets may be empty.
  ///
  /// (!) For the purposes of this function declarations are not considered to
  ///     be references. However, declarations can have references inside them,
  ///     e.g. 'namespace foo = std' references namespace 'std' and this
  ///     function will return the corresponding reference.
  llvm::SmallVector<ReferenceLoc, 2> explicitReference(DynTypedNode N) {
    if (auto *D = N.get<Decl>())
      return refInDecl(D);
    if (auto *S = N.get<Stmt>())
      return refInStmt(S);
    if (auto *NNSL = N.get<NestedNameSpecifierLoc>()) {
      // (!) 'DeclRelation::Alias' ensures we do not loose namespace aliases.
      return {ReferenceLoc{
          NNSL->getPrefix(), NNSL->getLocalBeginLoc(), false,
          explicitReferenceTargets(
              DynTypedNode::create(*NNSL->getNestedNameSpecifier()),
              DeclRelation::Alias)}};
    }
    if (const TypeLoc *TL = N.get<TypeLoc>())
      return refInTypeLoc(*TL);
    if (const CXXCtorInitializer *CCI = N.get<CXXCtorInitializer>()) {
      // Other type initializers (e.g. base initializer) are handled by visiting
      // the typeLoc.
      if (CCI->isAnyMemberInitializer()) {
        return {ReferenceLoc{NestedNameSpecifierLoc(),
                             CCI->getMemberLocation(),
                             /*IsDecl=*/false,
                             {CCI->getAnyMember()}}};
      }
    }
    // We do not have location information for other nodes (QualType, etc)
    return {};
  }

  void visitNode(DynTypedNode N) {
    for (const auto &R : explicitReference(N))
      reportReference(R, N);
  }

  void reportReference(const ReferenceLoc &Ref, DynTypedNode N) {
    // Our promise is to return only references from the source code. If we lack
    // location information, skip these nodes.
    // Normally this should not happen in practice, unless there are bugs in the
    // traversals or users started the traversal at an implicit node.
    if (Ref.NameLoc.isInvalid()) {
      dlog("invalid location at node {0}", nodeToString(N));
      return;
    }
    Out(Ref);
  }

  llvm::function_ref<void(ReferenceLoc)> Out;
  /// TypeLocs starting at these locations must be skipped, see
  /// TraverseElaboratedTypeSpecifierLoc for details.
  llvm::DenseSet</*SourceLocation*/ unsigned> TypeLocsToSkip;
};
} // namespace

void findExplicitReferences(const Stmt *S,
                            llvm::function_ref<void(ReferenceLoc)> Out) {
  assert(S);
  ExplicitReferenceCollector(Out).TraverseStmt(const_cast<Stmt *>(S));
}
void findExplicitReferences(const Decl *D,
                            llvm::function_ref<void(ReferenceLoc)> Out) {
  assert(D);
  ExplicitReferenceCollector(Out).TraverseDecl(const_cast<Decl *>(D));
}
void findExplicitReferences(const ASTContext &AST,
                            llvm::function_ref<void(ReferenceLoc)> Out) {
  ExplicitReferenceCollector(Out).TraverseAST(const_cast<ASTContext &>(AST));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, DeclRelation R) {
  switch (R) {
#define REL_CASE(X)                                                            \
  case DeclRelation::X:                                                        \
    return OS << #X;
    REL_CASE(Alias);
    REL_CASE(Underlying);
    REL_CASE(TemplateInstantiation);
    REL_CASE(TemplatePattern);
#undef REL_CASE
  }
  llvm_unreachable("Unhandled DeclRelation enum");
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, DeclRelationSet RS) {
  const char *Sep = "";
  for (unsigned I = 0; I < RS.S.size(); ++I) {
    if (RS.S.test(I)) {
      OS << Sep << static_cast<DeclRelation>(I);
      Sep = "|";
    }
  }
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, ReferenceLoc R) {
  // note we cannot print R.NameLoc without a source manager.
  OS << "targets = {";
  bool First = true;
  for (const NamedDecl *T : R.Targets) {
    if (!First)
      OS << ", ";
    else
      First = false;
    OS << printQualifiedName(*T) << printTemplateSpecializationArgs(*T);
  }
  OS << "}";
  if (R.Qualifier) {
    OS << ", qualifier = '";
    R.Qualifier.getNestedNameSpecifier()->print(OS,
                                                PrintingPolicy(LangOptions()));
    OS << "'";
  }
  if (R.IsDecl)
    OS << ", decl";
  return OS;
}

} // namespace clangd
} // namespace clang
