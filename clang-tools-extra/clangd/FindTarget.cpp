//===--- FindTarget.cpp - What does an AST node refer to? -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindTarget.h"
#include "AST.h"
#include "Logger.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace {

LLVM_ATTRIBUTE_UNUSED std::string
nodeToString(const ast_type_traits::DynTypedNode &N) {
  std::string S = N.getNodeKind().asStringRef();
  {
    llvm::raw_string_ostream OS(S);
    OS << ": ";
    N.print(OS, PrintingPolicy(LangOptions()));
  }
  std::replace(S.begin(), S.end(), '\n', ' ');
  return S;
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
// e.g. template<typename T> int foo() { return std::vector<T>().size(); }
// formally size() is unresolved, but the primary template is a good guess.
// This affects:
//  - DependentTemplateSpecializationType,
//  - DependentScopeMemberExpr
//  - DependentScopeDeclRefExpr
//  - DependentNameType
struct TargetFinder {
  using RelSet = DeclRelationSet;
  using Rel = DeclRelation;
  llvm::SmallDenseMap<const Decl *, RelSet> Decls;
  RelSet Flags;

  static const Decl *getTemplatePattern(const Decl *D) {
    if (const CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(D)) {
      return CRD->getTemplateInstantiationPattern();
    } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      return FD->getTemplateInstantiationPattern();
    } else if (auto *VD = dyn_cast<VarDecl>(D)) {
      // Hmm: getTIP returns its arg if it's not an instantiation?!
      VarDecl *T = VD->getTemplateInstantiationPattern();
      return (T == D) ? nullptr : T;
    } else if (const auto *ED = dyn_cast<EnumDecl>(D)) {
      return ED->getInstantiatedFromMemberEnum();
    } else if (isa<FieldDecl>(D) || isa<TypedefNameDecl>(D)) {
      const auto *ND = cast<NamedDecl>(D);
      if (const DeclContext *Parent = dyn_cast_or_null<DeclContext>(
              getTemplatePattern(llvm::cast<Decl>(ND->getDeclContext()))))
        for (const NamedDecl *BaseND : Parent->lookup(ND->getDeclName()))
          if (!BaseND->isImplicit() && BaseND->getKind() == ND->getKind())
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

  template <typename T> void debug(T &Node, RelSet Flags) {
    dlog("visit [{0}] {1}", Flags,
         nodeToString(ast_type_traits::DynTypedNode::create(Node)));
  }

  void report(const Decl *D, RelSet Flags) {
    dlog("--> [{0}] {1}", Flags,
         nodeToString(ast_type_traits::DynTypedNode::create(*D)));
    Decls[D] |= Flags;
  }

public:
  void add(const Decl *D, RelSet Flags) {
    if (!D)
      return;
    debug(*D, Flags);
    if (const UsingDirectiveDecl *UDD = llvm::dyn_cast<UsingDirectiveDecl>(D))
      D = UDD->getNominatedNamespaceAsWritten();

    if (const TypedefNameDecl *TND = dyn_cast<TypedefNameDecl>(D)) {
      add(TND->getUnderlyingType(), Flags | Rel::Underlying);
      Flags |= Rel::Alias; // continue with the alias.
    } else if (const UsingDecl *UD = dyn_cast<UsingDecl>(D)) {
      for (const UsingShadowDecl *S : UD->shadows())
        add(S->getUnderlyingDecl(), Flags | Rel::Underlying);
      Flags |= Rel::Alias; // continue with the alias.
    } else if (const auto *NAD = dyn_cast<NamespaceAliasDecl>(D)) {
      add(NAD->getUnderlyingDecl(), Flags | Rel::Underlying);
      Flags |= Rel::Alias; // continue with the alias
    } else if (const UsingShadowDecl *USD = dyn_cast<UsingShadowDecl>(D)) {
      // Include the using decl, but don't traverse it. This may end up
      // including *all* shadows, which we don't want.
      report(USD->getUsingDecl(), Flags | Rel::Alias);
      // Shadow decls are synthetic and not themselves interesting.
      // Record the underlying decl instead, if allowed.
      D = USD->getTargetDecl();
      Flags |= Rel::Underlying; // continue with the underlying decl.
    }

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
      void VisitDecltypeType(const DecltypeType *DTT) {
        Outer.add(DTT->getUnderlyingType(), Flags | Rel::Underlying);
      }
      void VisitDeducedType(const DeducedType *DT) {
        // FIXME: In practice this doesn't work: the AutoType you find inside
        // TypeLoc never has a deduced type. https://llvm.org/PR42914
        Outer.add(DT->getDeducedType(), Flags | Rel::Underlying);
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
};

} // namespace

llvm::SmallVector<std::pair<const Decl *, DeclRelationSet>, 1>
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

  return {Finder.Decls.begin(), Finder.Decls.end()};
}

llvm::SmallVector<const Decl *, 1>
targetDecl(const ast_type_traits::DynTypedNode &N, DeclRelationSet Mask) {
  llvm::SmallVector<const Decl *, 1> Result;
  for (const auto &Entry : allTargetDecls(N))
    if (!(Entry.second & ~Mask))
      Result.push_back(Entry.first);
  return Result;
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

} // namespace clangd
} // namespace clang
