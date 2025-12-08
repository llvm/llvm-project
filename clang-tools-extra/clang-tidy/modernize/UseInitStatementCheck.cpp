//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseInitStatementCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm> // for std::adjacent_find
#include <cctype>
#include <string>
#include <utility>

using namespace clang::ast_matchers;
using namespace clang::tidy::utils::lexer;

using clang::ast_matchers::internal::Matcher;
using clang::ast_matchers::internal::VariadicDynCastAllOfMatcher;

namespace clang::tidy::modernize {

namespace {

// Matches CompoundStmt that contains a PrevStmt immediately followed by
// NextStmt
// FIXME: use hasAdjSubstatements, see
// https://github.com/llvm/llvm-project/pull/169965
AST_MATCHER_P2(CompoundStmt, hasAdjacentStmts, Matcher<Stmt>, DeclMatcher,
               Matcher<Stmt>, StmtMatcher) {
  const auto Statements = Node.body();

  return std::adjacent_find(
             Statements.begin(), Statements.end(),
             [&](const Stmt *PrevStmt, const Stmt *NextStmt) {
               clang::ast_matchers::internal::BoundNodesTreeBuilder PrevBuilder;
               if (!DeclMatcher.matches(*PrevStmt, Finder, &PrevBuilder))
                 return false;

               clang::ast_matchers::internal::BoundNodesTreeBuilder NextBuilder;
               NextBuilder.addMatch(PrevBuilder);
               if (!StmtMatcher.matches(*NextStmt, Finder, &NextBuilder))
                 return false;

               Builder->addMatch(NextBuilder);
               return true;
             }) != Statements.end();
}

AST_MATCHER(Decl, isTemplate) { return Node.isTemplated(); }

AST_MATCHER(VarDecl, isUsed) {
  if (Node.getType().isConstQualified())
    return true; // FIXME: implement proper "used" check for consts
  return Node.isUsed();
}

// got from implementation of memberHasSameNameAsBoundNode mather
AST_MATCHER_P(VarDecl, hasSameNameAsBoundNode, std::string, BindingID) {
  auto VarName = Node.getNameAsString();

  return Builder->removeBindings(
      [this,
       VarName](const clang::ast_matchers::internal::BoundNodesMap &Nodes) {
        const DynTypedNode &BN = Nodes.getNode(this->BindingID);
        if (const auto *ND = BN.get<NamedDecl>()) {
          if (!isa<BindingDecl, VarDecl>(ND))
            return true;
          return ND->getName() != VarName;
        }
        return true;
      });
}

} // namespace

UseInitStatementCheck::UseInitStatementCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreConditionVariableStatements(
          Options.get("IgnoreConditionVariableStatements", false)) {}

void UseInitStatementCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreConditionVariableStatements",
                IgnoreConditionVariableStatements);
}

static Matcher<Stmt> callByRef(const Matcher<Decl> &VarOrBindingNodeMatcher) {
  const auto ArgMatcher = declRefExpr(to(VarOrBindingNodeMatcher));
  const auto ParamMatcher = parmVarDecl(hasType(referenceType()));

  return anyOf(
      callExpr(forEachArgumentWithParam(ArgMatcher, ParamMatcher)),
      cxxConstructExpr(forEachArgumentWithParam(ArgMatcher, ParamMatcher)));
}

static Matcher<VarDecl> hasInitializerWithLifetimeExtension() {
  return hasInitializer(expr(hasDescendant(expr(materializeTemporaryExpr()))));
}

static Matcher<DeclStmt> includingLifetimeExtension() {
  return has(varDecl(hasInitializerWithLifetimeExtension()));
}

static Matcher<DeclStmt> excludingLifetimeExtension() {
  return unless(includingLifetimeExtension());
}

// Complete matcher to detect stealing cases, preventing
// the checker from generating code that could result in dangling references,
// such as: `int* p; if (int v;cond) { p=&v; } use(p);`
static Matcher<Stmt> hasStealingMatcher() {
  const auto IsStealingViaPointer =
      hasParent(unaryOperator(hasOperatorName("&")));
  const auto HasLTE = hasInitializerWithLifetimeExtension();
  const auto HasLTEForBind =
      hasParent(declStmt(has(varDecl(equalsBoundNode("singleVar"), HasLTE))));
  const auto IsNonreferenceType = unless(hasType(referenceType()));
  const auto BoundVar =
      varDecl(equalsBoundNode("singleVar"), anyOf(IsNonreferenceType, HasLTE));
  const auto BoundBind = bindingDecl(
      equalsBoundNode("bindingDecl"),
      hasParent(decompositionDecl(anyOf(IsNonreferenceType, HasLTEForBind))));
  const auto VarOrBindingNode = anyOf(BoundVar, BoundBind);
  const auto StealingAsThis =
      anyOf(hasParent(memberExpr()),
            hasParent(implicitCastExpr(hasCastKind(CK_NoOp),
                                       hasParent(memberExpr()))));
  const auto HasStealingByPointer =
      hasDescendant(declRefExpr(to(VarOrBindingNode), IsStealingViaPointer));
  const auto HasStealingByReference =
      hasDescendant(callByRef(VarOrBindingNode));
  const auto HasStealingAsThis =
      hasDescendant(declRefExpr(to(VarOrBindingNode), StealingAsThis));
  return anyOf(HasStealingByPointer, HasStealingByReference, HasStealingAsThis);
}

static Matcher<Stmt> hasConflictMatcher() {
  return hasDescendant(varDecl(anyOf(hasSameNameAsBoundNode("singleVar"),
                                     hasSameNameAsBoundNode("bindingDecl")))
                           .bind("conflict"));
}

static Matcher<Stmt>
compoundStmtMatcher(const Matcher<Stmt> &StmtMatcher, StringRef StmtName,
                    const Matcher<Stmt> &PrevStmtMatcher,
                    const Matcher<Stmt> &RefToBoundMatcher) {
  const auto NoOtherVarRefs =
      unless(has(stmt(unless(equalsBoundNode(StmtName.str())),
                      hasDescendant(RefToBoundMatcher))));
  return compoundStmt(unless(isExpansionInSystemHeader()),
                      unless(hasAncestor(functionDecl(isTemplate()))),
                      hasAdjacentStmts(PrevStmtMatcher, StmtMatcher),
                      NoOtherVarRefs)
      .bind("compound");
}

// Complete matcher for if/switch statements that can be refactored to use
// init statements.
// Usage: compoundStmtMatcher(ifStmt, "ifStmt", declStmt(),
//                            declRefExpr(to(varDecl(equalsBoundNode("singleVar")))))
template <typename IfOrSwitchStmt>
static Matcher<Stmt> compoundStmtMatcher(
    const VariadicDynCastAllOfMatcher<Stmt, IfOrSwitchStmt> &StmtMatcher,
    StringRef StmtName, const Matcher<Stmt> &PrevStmtMatcher,
    const Matcher<Stmt> &RefToBoundMatcher) {
  const auto StmtWithCondition =
      StmtMatcher(unless(hasInitStatement(anything())),
                  unless(hasStealingMatcher()),
                  hasCondition(expr().bind("condition")),
                  optionally(hasConflictMatcher()),
                  optionally(hasConditionVariableStatement(
                      declStmt().bind("condDeclStmt"))))
          .bind(StmtName);
  return compoundStmtMatcher(StmtWithCondition, StmtName, PrevStmtMatcher,
                             RefToBoundMatcher);
}

template <bool IsDecomposition = false>
static auto forBuiltinTypes(const Matcher<ReferenceType> &ConditionForReference)
    -> std::conditional_t<IsDecomposition, Matcher<DecompositionDecl>,
                          Matcher<Stmt>> {
  const auto AllowedTypeMatcher = qualType(unless(
      anyOf(hasCanonicalType(builtinType()), hasCanonicalType(pointerType()),
            hasCanonicalType(referenceType(ConditionForReference)))));

  if constexpr (IsDecomposition) {
    const auto NonReferenceVarDecl =
        varDecl(hasType(qualType(unless(hasCanonicalType(referenceType())))));
    const auto BindingWithAllowedType =
        bindingDecl(hasType(AllowedTypeMatcher));
    const auto DecompositionWithAllowedBinding =
        unless(has(BindingWithAllowedType));
    const auto VarDeclInParent =
        hasParent(declStmt(unless(has(NonReferenceVarDecl))));
    return anyOf(DecompositionWithAllowedBinding, VarDeclInParent);
  } else {
    const auto VarDeclWithAllowedType = varDecl(hasType(AllowedTypeMatcher));
    return unless(has(VarDeclWithAllowedType));
  }
}

// Registers matchers for if/switch statements with regular variable
// declarations.
void UseInitStatementCheck::registerVariableDeclMatchers(MatchFinder *Finder) {
  // Matchers for variable declarations
  const auto SingleVarDecl = varDecl(isUsed()).bind("singleVar");
  const auto RefToBoundVarDecl =
      declRefExpr(to(varDecl(equalsBoundNode("singleVar"))));

  // Matchers for declaration statements that precede if/switch
  const auto PrevDeclStmt =
      declStmt(forEach(SingleVarDecl), forBuiltinTypes(anything()),
               excludingLifetimeExtension())
          .bind("prevDecl");
  const auto PrevDeclStmtLTE =
      declStmt(forEach(SingleVarDecl), forBuiltinTypes(pointee(builtinType())),
               includingLifetimeExtension())
          .bind("prevDecl");
  const auto PrevDeclStmtMatcher = anyOf(PrevDeclStmt, PrevDeclStmtLTE);

  Finder->addMatcher(compoundStmtMatcher(ifStmt, "ifStmt", PrevDeclStmtMatcher,
                                         RefToBoundVarDecl),
                     this);
  Finder->addMatcher(compoundStmtMatcher(switchStmt, "switchStmt",
                                         PrevDeclStmtMatcher,
                                         RefToBoundVarDecl),
                     this);
}

// Registers matchers for if/switch statements with structured bindings.
void UseInitStatementCheck::registerStructuredBindingMatchers(
    MatchFinder *Finder) {
  // Matchers for variable declarations
  const auto SingleVarDecl = varDecl().bind("singleVar");

  const auto DecompositionDecl =
      decompositionDecl(forEach(bindingDecl().bind("bindingDecl")),
                        forBuiltinTypes<true>(anything()));
  const auto DecompositionDeclLTE =
      decompositionDecl(forEach(bindingDecl().bind("bindingDecl")),
                        forBuiltinTypes<true>(pointee(builtinType())));
  const auto PrevDecomposeStmt =
      declStmt(has(DecompositionDecl), has(SingleVarDecl),
               excludingLifetimeExtension())
          .bind("prevDecl");
  const auto PrevDecomposeStmtLTE =
      declStmt(has(DecompositionDeclLTE), has(SingleVarDecl),
               includingLifetimeExtension())
          .bind("prevDecl");
  const auto PrevDecomposeStmtMatcher =
      anyOf(PrevDecomposeStmt, PrevDecomposeStmtLTE);
  const auto RefToBoundBindDecl =
      declRefExpr(to(bindingDecl(equalsBoundNode("bindingDecl"))));

  Finder->addMatcher(compoundStmtMatcher(ifStmt, "ifStmt",
                                         PrevDecomposeStmtMatcher,
                                         RefToBoundBindDecl),
                     this);
  Finder->addMatcher(compoundStmtMatcher(switchStmt, "switchStmt",
                                         PrevDecomposeStmtMatcher,
                                         RefToBoundBindDecl),
                     this);
}

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  registerVariableDeclMatchers(Finder);
  registerStructuredBindingMatchers(Finder);
}

static std::string extractDeclStmtText(const DeclStmt *PrevDecl,
                                       const SourceManager *SM,
                                       const LangOptions &LangOpts) {
  const SourceRange CuttingRange = PrevDecl->getSourceRange();
  const CharSourceRange DeclCharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(CuttingRange), *SM, LangOpts);
  const StringRef DeclStmtText =
      DeclCharRange.isInvalid()
          ? ""
          : Lexer::getSourceText(DeclCharRange, *SM, LangOpts);
  return DeclStmtText.empty() ? "" : DeclStmtText.trim().str() + " ";
}

int UseInitStatementCheck::getKindOfMatchedDeclStmt(
    const MatchFinder::MatchResult &Result) const {
  const auto *BD = Result.Nodes.getNodeAs<BindingDecl>("bindingDecl");
  if (BD)
    return 0;

  const auto *D = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  return (llvm::size(D->decls()) > 1) ? 1 : 2;
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<Stmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<Stmt>("switchStmt");
  const auto *Conflict = Result.Nodes.getNodeAs<VarDecl>("conflict");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *CondDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("condDeclStmt");
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  const auto *Statement = If ? If : Switch;

  if (!PrevDecl || !Condition || !Compound || !Statement)
    return;

  if (IgnoreConditionVariableStatements && CondDeclStmt)
    return;

  // Don't emit warnings for statements inside macro expansions
  if (Statement->getBeginLoc().isMacroID())
    return;

  // Don't emit warnings if there are preprocessor conditionals between
  // the variable declaration and the if/switch statement
  if (const SourceRange Range(PrevDecl->getSourceRange().getBegin(),
                              Compound->getSourceRange().getEnd());
      Range.isInvalid() || rangeContainsExpansionsOrDirectives(
                               Range, *Result.SourceManager, getLangOpts()))
    return;

  auto Diag = diag(PrevDecl->getBeginLoc(),
                   "%select{structured binding|multiple variable|variable %1}0 "
                   "declaration "
                   "before %select{if|switch}2 statement could be moved into "
                   "%select{if|switch}2 init statement")
              << getKindOfMatchedDeclStmt(Result)
              << llvm::dyn_cast<VarDecl>(*PrevDecl->decl_begin()) << !If;

  const std::string NewInitStmtOpt =
      extractDeclStmtText(PrevDecl, Result.SourceManager, getLangOpts());
  const bool CanFix = !NewInitStmtOpt.empty() && !Conflict;

  if (CanFix) {
    const SourceRange RemovalRange = PrevDecl->getSourceRange();
    // Determine the insertion point: if the condition contains a declaration,
    // insert before that declaration; otherwise insert before the condition.
    const SourceLocation InsertionLoc =
        CondDeclStmt ? CondDeclStmt->getBeginLoc() : Condition->getBeginLoc();

    Diag << FixItHint::CreateRemoval(RemovalRange)
         << FixItHint::CreateInsertion(InsertionLoc, NewInitStmtOpt);
  }
}

} // namespace clang::tidy::modernize
