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

// FIXME: support other std:: types, like std::vector
const auto DefaultSafeTypes =
    "-*,::std::*string,::std::*string_view,::boost::*string,::boost::*string_"
    "view,::boost::*string_ref";

} // namespace

UseInitStatementCheck::UseInitStatementCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", true)),
      IgnoreConditionVariableStatements(
          Options.get("IgnoreConditionVariableStatements", false)),
      SafeTypes(Options.get("SafeTypes", DefaultSafeTypes)),
      SafeDestructorTypesGlobList(SafeTypes) {}

void UseInitStatementCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "IgnoreConditionVariableStatements",
                IgnoreConditionVariableStatements);
  Options.store(Opts, "SafeTypes", SafeTypes);
}

static Matcher<Stmt> callByRef(Matcher<Decl> VarOrBindingNodeMatcher,
                               StringRef Name) {
  const auto argMatcher = declRefExpr(to(VarOrBindingNodeMatcher)).bind(Name);
  const auto paramMatcher = parmVarDecl(hasType(referenceType()));

  return anyOf(
      callExpr(forEachArgumentWithParam(argMatcher, paramMatcher)),
      cxxConstructExpr(forEachArgumentWithParam(argMatcher, paramMatcher)));
}

static Matcher<Decl> bindToNonreferenceMatcher() {
  const auto IsNonreferenceType = unless(hasType(referenceType()));
  const auto BoundVar =
      varDecl(equalsBoundNode("singleVar"), IsNonreferenceType);
  const auto BoundBind =
      bindingDecl(equalsBoundNode("bindingDecl"),
                  hasParent(decompositionDecl(IsNonreferenceType)));
  return anyOf(BoundVar, BoundBind);
}

// Complete matcher to detect stealing cases, preventing
// the checker from generating code that could result in dangling references,
// such as: `int* p; if (int v;cond) { p=&v; } use(p);`
static Matcher<Stmt> hasStealingMatcher() {
  const auto IsStealingViaPointer =
      hasParent(unaryOperator(hasOperatorName("&")));
  const auto VarOrBindingNode = bindToNonreferenceMatcher();
  const auto HasStealingByPointer = hasDescendant(
      declRefExpr(to(VarOrBindingNode), IsStealingViaPointer).bind("stealing"));
  const auto HasStealingByReference =
      hasDescendant(callByRef(VarOrBindingNode, "stealing"));
  return anyOf(HasStealingByPointer, HasStealingByReference);
}

// Matcher to detect when a variable or binding is used as 'this' in a member
// expression, such as: `A v; if (cond) { v.store_this_somewhere(); }
// use_last_stored_this();`
static Matcher<Stmt> hasStealingAsThisMatcher() {
  const auto VarOrBindingNode = bindToNonreferenceMatcher();
  const auto StealingAsThis =
      anyOf(hasParent(memberExpr()),
            hasParent(implicitCastExpr(hasCastKind(CK_NoOp),
                                       hasParent(memberExpr()))));
  return hasDescendant(
      declRefExpr(to(VarOrBindingNode), StealingAsThis).bind("stealingAsThis"));
}

static Matcher<Stmt> hasConflictMatcher() {
  return hasDescendant(varDecl(anyOf(hasSameNameAsBoundNode("singleVar"),
                                     hasSameNameAsBoundNode("bindingDecl")))
                           .bind("conflict"));
}

static Matcher<Stmt> compoundStmtMatcher(Matcher<Stmt> StmtMatcher,
                                         StringRef StmtName,
                                         Matcher<Stmt> PrevStmtMatcher,
                                         Matcher<Stmt> RefToBoundMatcher) {
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
    VariadicDynCastAllOfMatcher<Stmt, IfOrSwitchStmt> StmtMatcher,
    StringRef StmtName, Matcher<Stmt> PrevStmtMatcher,
    Matcher<Stmt> RefToBoundMatcher) {
  const auto StmtWithCondition =
      StmtMatcher(unless(hasInitStatement(anything())),
                  hasCondition(expr().bind("condition")),
                  optionally(hasConflictMatcher()),
                  optionally(hasStealingMatcher()),
                  optionally(hasStealingAsThisMatcher()),
                  optionally(hasConditionVariableStatement(
                      declStmt().bind("condDeclStmt"))))
          .bind(StmtName);
  return compoundStmtMatcher(StmtWithCondition, StmtName, PrevStmtMatcher,
                             RefToBoundMatcher);
}

static Matcher<VarDecl> hasDestructorMatcher() {
  const auto ClassWithDtorDecl =
      cxxRecordDecl(hasMethod(cxxDestructorDecl().bind("dtorDecl")));
  const auto ClassWithDtorType =
      hasCanonicalType(hasDeclaration(ClassWithDtorDecl));
  const auto ArrayOfClassWithDtor =
      hasType(arrayType(hasElementType(ClassWithDtorType)));
  return anyOf(hasType(ClassWithDtorType), ArrayOfClassWithDtor);
}

static Matcher<VarDecl> hasLifetimeExtensionMatcher() {
  return hasInitializer(
      expr(hasDescendant(expr(materializeTemporaryExpr().bind("lte")))));
}

static std::pair<Matcher<Decl>, Matcher<Decl>> getSingleVarDeclMatchers() {
  const auto HasDtor = hasDestructorMatcher();
  const auto HasLTE = hasLifetimeExtensionMatcher();

  return {varDecl(HasDtor, optionally(HasLTE), isUsed()).bind("singleVar"),
          varDecl(optionally(HasLTE), isUsed()).bind("singleVar")};
}

// Registers matchers for if/switch statements with regular variable
// declarations.
void UseInitStatementCheck::registerVariableDeclMatchers(MatchFinder *Finder) {
  // Matchers for variable declarations
  const auto [SingleVarDeclWithDtor, SingleVarDecl] =
      getSingleVarDeclMatchers();
  const auto RefToBoundVarDecl =
      declRefExpr(to(varDecl(equalsBoundNode("singleVar"))));

  // Matchers for declaration statements that precede if/switch
  const auto PrevDeclStmtWithDtor =
      declStmt(forEach(SingleVarDeclWithDtor)).bind("prevDecl");
  const auto PrevDeclStmt = declStmt(forEach(SingleVarDecl)).bind("prevDecl");
  const auto PrevDeclStmtMatcher = anyOf(PrevDeclStmtWithDtor, PrevDeclStmt);

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
  const auto [SingleVarDeclWithDtor, SingleVarDecl] =
      getSingleVarDeclMatchers();

  const auto DecompositionDecl =
      decompositionDecl(forEach(bindingDecl().bind("bindingDecl")));
  const auto PrevDecomposeStmtWithDtor =
      declStmt(has(DecompositionDecl), has(SingleVarDeclWithDtor))
          .bind("prevDecl");
  const auto PrevDecomposeStmt =
      declStmt(has(DecompositionDecl), has(SingleVarDecl)).bind("prevDecl");
  const auto PrevDecomposeStmtMatcher =
      anyOf(PrevDecomposeStmtWithDtor, PrevDecomposeStmt);
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

static bool isLastInCompound(const Stmt *S, const CompoundStmt *P) {
  return !P->body_empty() && P->body_back() == S;
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

bool UseInitStatementCheck::isSingleVarWithSafeDestructor(
    const MatchFinder::MatchResult &Result) const {
  if (SafeTypes.empty())
    return false;

  const auto *SingleVar = Result.Nodes.getNodeAs<VarDecl>("singleVar");
  if (!SingleVar)
    return false;

  const QualType VarType = SingleVar->getType();
  const PrintingPolicy Policy(Result.Context->getLangOpts());
  const std::string TypeName = TypeName::getFullyQualifiedName(
      VarType.getUnqualifiedType(), *Result.Context, Policy,
      /*WithGlobalNsPrefix=*/true);

  // Check if the type name matches any of the safe types using GlobList
  return SafeDestructorTypesGlobList.contains(TypeName);
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
  const auto *Dtor = Result.Nodes.getNodeAs<CXXDestructorDecl>("dtorDecl");
  const auto *LTE = Result.Nodes.getNodeAs<MaterializeTemporaryExpr>("lte");
  const auto *Conflict = Result.Nodes.getNodeAs<VarDecl>("conflict");
  const auto *Stealing = Result.Nodes.getNodeAs<DeclRefExpr>("stealing");
  const auto *StealingAsThis =
      Result.Nodes.getNodeAs<DeclRefExpr>("stealingAsThis");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *CondDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("condDeclStmt");
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  const auto *Statement = If ? If : Switch;

  if (!PrevDecl || !Condition || !Compound || !Statement)
    return;

  const bool IsLast = isLastInCompound(Statement, Compound);

  if (!StrictMode && IsLast)
    return;

  if (IgnoreConditionVariableStatements && CondDeclStmt)
    return;

  if (!IsLast) {
    const bool IsSafe = isSingleVarWithSafeDestructor(Result);
    if (Dtor && !IsSafe)
      return;

    if (Stealing || (StealingAsThis && !IsSafe))
      return;

    if (LTE)
      return;
  }

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
