//===--- DumpAST.cpp - Serialize clang AST to LSP -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DumpAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TextNodeDumper.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

using llvm::raw_ostream;
template <typename Print> std::string toString(const Print &C) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  C(OS);
  return std::move(OS.str());
}

bool isInjectedClassName(Decl *D) {
  if (const auto *CRD = llvm::dyn_cast<CXXRecordDecl>(D))
    return CRD->isInjectedClassName();
  return false;
}

class DumpVisitor : public RecursiveASTVisitor<DumpVisitor> {
  using Base = RecursiveASTVisitor<DumpVisitor>;

  const syntax::TokenBuffer &Tokens;
  const ASTContext &Ctx;

  // Pointers are into 'children' vector.
  // They remain valid because while a node is on the stack we only add
  // descendants, not siblings.
  std::vector<ASTNode *> Stack;

  // Generic logic used to handle traversal of all node kinds.

  template <typename T>
  bool traverseNodePre(llvm::StringRef Role, const T &Node) {
    if (Stack.empty()) {
      assert(Root.role.empty());
      Stack.push_back(&Root);
    } else {
      Stack.back()->children.emplace_back();
      Stack.push_back(&Stack.back()->children.back());
    }
    auto &N = *Stack.back();
    N.role = Role.str();
    N.kind = getKind(Node);
    N.detail = getDetail(Node);
    N.range = getRange(Node);
    N.arcana = getArcana(Node);
    return true;
  }
  bool traverseNodePost() {
    assert(!Stack.empty());
    Stack.pop_back();
    return true;
  }
  template <typename T, typename Callable>
  bool traverseNode(llvm::StringRef Role, const T &Node, const Callable &Body) {
    traverseNodePre(Role, Node);
    Body();
    return traverseNodePost();
  }

  // Range: most nodes have getSourceRange(), with a couple of exceptions.
  // We only return it if it's valid at both ends and there are no macros.

  template <typename T> std::optional<Range> getRange(const T &Node) {
    SourceRange SR = getSourceRange(Node);
    auto Spelled = Tokens.spelledForExpanded(Tokens.expandedTokens(SR));
    if (!Spelled)
      return std::nullopt;
    return halfOpenToRange(
        Tokens.sourceManager(),
        CharSourceRange::getCharRange(Spelled->front().location(),
                                      Spelled->back().endLocation()));
  }
  template <typename T, typename = decltype(std::declval<T>().getSourceRange())>
  SourceRange getSourceRange(const T &Node) {
    return Node.getSourceRange();
  }
  template <typename T,
            typename = decltype(std::declval<T *>()->getSourceRange())>
  SourceRange getSourceRange(const T *Node) {
    return Node->getSourceRange();
  }
  // TemplateName doesn't have a real Loc node type.
  SourceRange getSourceRange(const TemplateName &Node) { return SourceRange(); }
  // Attr just uses a weird method name. Maybe we should fix it instead?
  SourceRange getSourceRange(const Attr *Node) { return Node->getRange(); }

  // Kind is usually the class name, without the suffix ("Type" etc).
  // Where there's a set of variants instead, we use the 'Kind' enum values.

  std::string getKind(const Decl *D) { return D->getDeclKindName(); }
  std::string getKind(const Stmt *S) {
    std::string Result = S->getStmtClassName();
    if (llvm::StringRef(Result).ends_with("Stmt") ||
        llvm::StringRef(Result).ends_with("Expr"))
      Result.resize(Result.size() - 4);
    return Result;
  }
  std::string getKind(const TypeLoc &TL) {
    if (TL.getTypeLocClass() == TypeLoc::Qualified)
      return "Qualified";
    return TL.getType()->getTypeClassName();
  }
  std::string getKind(const TemplateArgumentLoc &TAL) {
    switch (TAL.getArgument().getKind()) {
#define TEMPLATE_ARGUMENT_KIND(X)                                              \
  case TemplateArgument::X:                                                    \
    return #X
      TEMPLATE_ARGUMENT_KIND(Null);
      TEMPLATE_ARGUMENT_KIND(NullPtr);
      TEMPLATE_ARGUMENT_KIND(Expression);
      TEMPLATE_ARGUMENT_KIND(Integral);
      TEMPLATE_ARGUMENT_KIND(Pack);
      TEMPLATE_ARGUMENT_KIND(Type);
      TEMPLATE_ARGUMENT_KIND(Declaration);
      TEMPLATE_ARGUMENT_KIND(Template);
      TEMPLATE_ARGUMENT_KIND(TemplateExpansion);
      TEMPLATE_ARGUMENT_KIND(StructuralValue);
#undef TEMPLATE_ARGUMENT_KIND
    }
    llvm_unreachable("Unhandled ArgKind enum");
  }
  std::string getKind(NestedNameSpecifierLoc NNSL) {
    switch (NNSL.getNestedNameSpecifier().getKind()) {
    case NestedNameSpecifier::Kind::Null:
      llvm_unreachable("unexpected null nested name specifier");
#define NNS_KIND(X)                                                            \
  case NestedNameSpecifier::Kind::X:                                           \
    return #X
      NNS_KIND(Namespace);
      NNS_KIND(Type);
      NNS_KIND(Global);
      NNS_KIND(MicrosoftSuper);
#undef NNS_KIND
    }
    llvm_unreachable("Unhandled SpecifierKind enum");
  }
  std::string getKind(const CXXCtorInitializer *CCI) {
    if (CCI->isBaseInitializer())
      return "BaseInitializer";
    if (CCI->isDelegatingInitializer())
      return "DelegatingInitializer";
    if (CCI->isAnyMemberInitializer())
      return "MemberInitializer";
    llvm_unreachable("Unhandled CXXCtorInitializer type");
  }
  std::string getKind(const TemplateName &TN) {
    switch (TN.getKind()) {
#define TEMPLATE_KIND(X)                                                       \
  case TemplateName::X:                                                        \
    return #X;
      TEMPLATE_KIND(Template);
      TEMPLATE_KIND(OverloadedTemplate);
      TEMPLATE_KIND(AssumedTemplate);
      TEMPLATE_KIND(QualifiedTemplate);
      TEMPLATE_KIND(DependentTemplate);
      TEMPLATE_KIND(SubstTemplateTemplateParm);
      TEMPLATE_KIND(SubstTemplateTemplateParmPack);
      TEMPLATE_KIND(UsingTemplate);
      TEMPLATE_KIND(DeducedTemplate);
#undef TEMPLATE_KIND
    }
    llvm_unreachable("Unhandled NameKind enum");
  }
  std::string getKind(const Attr *A) {
    switch (A->getKind()) {
#define ATTR(X)                                                                \
  case attr::X:                                                                \
    return #X;
#include "clang/Basic/AttrList.inc"
#undef ATTR
    }
    llvm_unreachable("Unhandled attr::Kind enum");
  }
  std::string getKind(const CXXBaseSpecifier &CBS) {
    // There aren't really any variants of CXXBaseSpecifier.
    // To avoid special cases in the API/UI, use public/private as the kind.
    return getAccessSpelling(CBS.getAccessSpecifier()).str();
  }
  std::string getKind(const ConceptReference *CR) {
    // Again there are no variants here.
    // Kind is "Concept", role is "reference"
    return "Concept";
  }

  // Detail is the single most important fact about the node.
  // Often this is the name, sometimes a "kind" enum like operators or casts.
  // We should avoid unbounded text, like dumping parameter lists.

  std::string getDetail(const Decl *D) {
    const auto *ND = dyn_cast<NamedDecl>(D);
    if (!ND || llvm::isa_and_nonnull<CXXConstructorDecl>(ND->getAsFunction()) ||
        isa<CXXDestructorDecl>(ND))
      return "";
    std::string Name = toString([&](raw_ostream &OS) { ND->printName(OS); });
    if (Name.empty())
      return "(anonymous)";
    return Name;
  }
  std::string getDetail(const Stmt *S) {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(S))
      return DRE->getNameInfo().getAsString();
    if (const auto *DSDRE = dyn_cast<DependentScopeDeclRefExpr>(S))
      return DSDRE->getNameInfo().getAsString();
    if (const auto *ME = dyn_cast<MemberExpr>(S))
      return ME->getMemberNameInfo().getAsString();
    if (const auto *CE = dyn_cast<CastExpr>(S))
      return CE->getCastKindName();
    if (const auto *BO = dyn_cast<BinaryOperator>(S))
      return BO->getOpcodeStr().str();
    if (const auto *UO = dyn_cast<UnaryOperator>(S))
      return UnaryOperator::getOpcodeStr(UO->getOpcode()).str();
    if (const auto *CCO = dyn_cast<CXXConstructExpr>(S))
      return CCO->getConstructor()->getNameAsString();
    if (const auto *CTE = dyn_cast<CXXThisExpr>(S)) {
      bool Const = CTE->getType()->getPointeeType().isLocalConstQualified();
      if (CTE->isImplicit())
        return Const ? "const, implicit" : "implicit";
      if (Const)
        return "const";
      return "";
    }
    if (isa<IntegerLiteral, FloatingLiteral, FixedPointLiteral,
            CharacterLiteral, ImaginaryLiteral, CXXBoolLiteralExpr>(S))
      return toString([&](raw_ostream &OS) {
        S->printPretty(OS, nullptr, Ctx.getPrintingPolicy());
      });
    if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(S))
      return MTE->isBoundToLvalueReference() ? "lvalue" : "rvalue";
    return "";
  }
  std::string getDetail(const TypeLoc &TL) {
    if (TL.getType().hasLocalQualifiers())
      return TL.getType().getLocalQualifiers().getAsString(
          Ctx.getPrintingPolicy());
    if (const auto *TT = dyn_cast<TagType>(TL.getTypePtr()))
      return getDetail(TT->getOriginalDecl());
    if (const auto *DT = dyn_cast<DeducedType>(TL.getTypePtr()))
      if (DT->isDeduced())
        return DT->getDeducedType().getAsString(Ctx.getPrintingPolicy());
    if (const auto *BT = dyn_cast<BuiltinType>(TL.getTypePtr()))
      return BT->getName(Ctx.getPrintingPolicy()).str();
    if (const auto *TTPT = dyn_cast<TemplateTypeParmType>(TL.getTypePtr()))
      return getDetail(TTPT->getDecl());
    if (const auto *TT = dyn_cast<TypedefType>(TL.getTypePtr()))
      return getDetail(TT->getDecl());
    return "";
  }
  std::string getDetail(NestedNameSpecifierLoc NNSL) {
    NestedNameSpecifier NNS = NNSL.getNestedNameSpecifier();
    if (NNS.getKind() != NestedNameSpecifier::Kind::Namespace)
      return "";
    return NNS.getAsNamespaceAndPrefix().Namespace->getNameAsString() + "::";
  }
  std::string getDetail(const CXXCtorInitializer *CCI) {
    if (FieldDecl *FD = CCI->getAnyMember())
      return getDetail(FD);
    if (TypeLoc TL = CCI->getBaseClassLoc())
      return getDetail(TL);
    return "";
  }
  std::string getDetail(const TemplateArgumentLoc &TAL) {
    if (TAL.getArgument().getKind() == TemplateArgument::Integral)
      return toString(TAL.getArgument().getAsIntegral(), 10);
    return "";
  }
  std::string getDetail(const TemplateName &TN) {
    return toString([&](raw_ostream &OS) {
      TN.print(OS, Ctx.getPrintingPolicy(), TemplateName::Qualified::None);
    });
  }
  std::string getDetail(const Attr *A) {
    return A->getAttrName() ? A->getNormalizedFullName() : A->getSpelling();
  }
  std::string getDetail(const CXXBaseSpecifier &CBS) {
    return CBS.isVirtual() ? "virtual" : "";
  }
  std::string getDetail(const ConceptReference *CR) {
    return CR->getNamedConcept()->getNameAsString();
  }

  /// Arcana is produced by TextNodeDumper, for the types it supports.

  template <typename Dump> std::string dump(const Dump &D) {
    return toString([&](raw_ostream &OS) {
      TextNodeDumper Dumper(OS, Ctx, /*ShowColors=*/false);
      D(Dumper);
    });
  }
  template <typename T> std::string getArcana(const T &N) {
    return dump([&](TextNodeDumper &D) { D.Visit(N); });
  }
  std::string getArcana(const NestedNameSpecifierLoc &NNS) { return ""; }
  std::string getArcana(const TemplateName &NNS) { return ""; }
  std::string getArcana(const CXXBaseSpecifier &CBS) { return ""; }
  std::string getArcana(const TemplateArgumentLoc &TAL) {
    return dump([&](TextNodeDumper &D) {
      D.Visit(TAL.getArgument(), TAL.getSourceRange());
    });
  }
  std::string getArcana(const TypeLoc &TL) {
    return dump([&](TextNodeDumper &D) { D.Visit(TL.getType()); });
  }

public:
  ASTNode Root;
  DumpVisitor(const syntax::TokenBuffer &Tokens, const ASTContext &Ctx)
      : Tokens(Tokens), Ctx(Ctx) {}

  // Override traversal to record the nodes we care about.
  // Generally, these are nodes with position information (TypeLoc, not Type).

  bool TraverseDecl(Decl *D) {
    return !D || isInjectedClassName(D) ||
           traverseNode("declaration", D, [&] { Base::TraverseDecl(D); });
  }
  bool TraverseTypeLoc(TypeLoc TL, bool TraverseQualifier = true) {
    return !TL || traverseNode("type", TL, [&] {
      Base::TraverseTypeLoc(TL, TraverseQualifier);
    });
  }
  bool TraverseTemplateName(const TemplateName &TN) {
    return traverseNode("template name", TN,
                        [&] { Base::TraverseTemplateName(TN); });
  }
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &TAL) {
    return traverseNode("template argument", TAL,
                        [&] { Base::TraverseTemplateArgumentLoc(TAL); });
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSL) {
    return !NNSL || traverseNode("specifier", NNSL, [&] {
      Base::TraverseNestedNameSpecifierLoc(NNSL);
    });
  }
  bool TraverseConstructorInitializer(CXXCtorInitializer *CCI) {
    return !CCI || traverseNode("constructor initializer", CCI, [&] {
      Base::TraverseConstructorInitializer(CCI);
    });
  }
  bool TraverseAttr(Attr *A) {
    return !A || traverseNode("attribute", A, [&] { Base::TraverseAttr(A); });
  }
  bool TraverseConceptReference(ConceptReference *C) {
    return !C || traverseNode("reference", C,
                              [&] { Base::TraverseConceptReference(C); });
  }
  bool TraverseCXXBaseSpecifier(const CXXBaseSpecifier &CBS) {
    return traverseNode("base", CBS,
                        [&] { Base::TraverseCXXBaseSpecifier(CBS); });
  }
  // Stmt is the same, but this form allows the data recursion optimization.
  bool dataTraverseStmtPre(Stmt *S) {
    return S && traverseNodePre(isa<Expr>(S) ? "expression" : "statement", S);
  }
  bool dataTraverseStmtPost(Stmt *X) { return traverseNodePost(); }

  // QualifiedTypeLoc is handled strangely in RecursiveASTVisitor: the derived
  // TraverseTypeLoc is not called for the inner UnqualTypeLoc.
  // This means we'd never see 'int' in 'const int'! Work around that here.
  // (The reason for the behavior is to avoid traversing the nested Type twice,
  // but we ignore TraverseType anyway).
  bool TraverseQualifiedTypeLoc(QualifiedTypeLoc QTL, bool TraverseQualifier) {
    return TraverseTypeLoc(QTL.getUnqualifiedLoc());
  }
  // Uninteresting parts of the AST that don't have locations within them.
  bool TraverseNestedNameSpecifier(NestedNameSpecifier) { return true; }
  bool TraverseType(QualType) { return true; }

  // OpaqueValueExpr blocks traversal, we must explicitly traverse it.
  bool TraverseOpaqueValueExpr(OpaqueValueExpr *E) {
    return TraverseStmt(E->getSourceExpr());
  }
  // We only want to traverse the *syntactic form* to understand the selection.
  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    return TraverseStmt(E->getSyntacticForm());
  }
};

} // namespace

ASTNode dumpAST(const DynTypedNode &N, const syntax::TokenBuffer &Tokens,
                const ASTContext &Ctx) {
  DumpVisitor V(Tokens, Ctx);
  // DynTypedNode only works with const, RecursiveASTVisitor only non-const :-(
  if (const auto *D = N.get<Decl>())
    V.TraverseDecl(const_cast<Decl *>(D));
  else if (const auto *S = N.get<Stmt>())
    V.TraverseStmt(const_cast<Stmt *>(S));
  else if (const auto *NNSL = N.get<NestedNameSpecifierLoc>())
    V.TraverseNestedNameSpecifierLoc(
        *const_cast<NestedNameSpecifierLoc *>(NNSL));
  else if (const auto *NNS = N.get<NestedNameSpecifier>())
    V.TraverseNestedNameSpecifier(*NNS);
  else if (const auto *TL = N.get<TypeLoc>())
    V.TraverseTypeLoc(*const_cast<TypeLoc *>(TL));
  else if (const auto *QT = N.get<QualType>())
    V.TraverseType(*const_cast<QualType *>(QT));
  else if (const auto *CCI = N.get<CXXCtorInitializer>())
    V.TraverseConstructorInitializer(const_cast<CXXCtorInitializer *>(CCI));
  else if (const auto *TAL = N.get<TemplateArgumentLoc>())
    V.TraverseTemplateArgumentLoc(*const_cast<TemplateArgumentLoc *>(TAL));
  else if (const auto *CBS = N.get<CXXBaseSpecifier>())
    V.TraverseCXXBaseSpecifier(*const_cast<CXXBaseSpecifier *>(CBS));
  else if (const auto *CR = N.get<ConceptReference>())
    V.TraverseConceptReference(const_cast<ConceptReference *>(CR));
  else
    elog("dumpAST: unhandled DynTypedNode kind {0}",
         N.getNodeKind().asStringRef());
  return std::move(V.Root);
}

} // namespace clangd
} // namespace clang
