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
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <cctype>
#include <optional>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

// Matches CompoundStmt that contains a PrevStmt immediately followed by
// NextStmt
AST_MATCHER_P2(CompoundStmt, hasAdjacentStmts,
               ast_matchers::internal::Matcher<Stmt>, DeclMatcher,
               ast_matchers::internal::Matcher<Stmt>, StmtMatcher) {
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

} // namespace

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  const auto ClassWithDtorDecl = cxxRecordDecl(hasMethod(cxxDestructorDecl()));
  const auto ClassWithDtorType =
      hasCanonicalType(hasDeclaration(ClassWithDtorDecl));
  const auto ArrayOfClassWithDtor =
      hasType(arrayType(hasElementType(ClassWithDtorType)));

  const auto SingleVarDecl =
      varDecl(unless(anyOf(hasType(ClassWithDtorType), ArrayOfClassWithDtor)))
          .bind("singleVar");
  const auto RefToBoundVarDecl =
      declRefExpr(to(varDecl(equalsBoundNode("singleVar"))));

  // Matcher for the declaration statement that precedes the if/switch
  const auto PrevDeclStmt = declStmt(forEach(SingleVarDecl)).bind("prevDecl");

  // Matcher for a condition that references the variable from prevDecl
  const auto ConditionWithVarRef =
      expr(forEachDescendant(RefToBoundVarDecl)).bind("condition");

  // Helper to create a complete matcher for if/switch statements
  const auto MakeCompoundMatcher = [&](const auto &StmtMatcher,
                                       const std::string &StmtName) {
    const auto StmtMatcherWithCondition =
        StmtMatcher(unless(hasInitStatement(anything())),
                    hasCondition(ConditionWithVarRef))
            .bind(StmtName);

    // Ensure the variable is not referenced elsewhere in the compound statement
    const auto NoOtherVarRefs = unless(has(stmt(
        unless(equalsBoundNode(StmtName)), hasDescendant(RefToBoundVarDecl))));

    return compoundStmt(
        unless(isInTemplateInstantiation()),
        hasAdjacentStmts(PrevDeclStmt, StmtMatcherWithCondition),
        NoOtherVarRefs);
  };

  Finder->addMatcher(MakeCompoundMatcher(ifStmt, "ifStmt"), this);
  Finder->addMatcher(MakeCompoundMatcher(switchStmt, "switchStmt"), this);
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switchStmt");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const Stmt *Statement = If ? static_cast<const Stmt *>(If) : Switch;

  if (!PrevDecl || !Condition || !Statement)
    return;

  const SourceRange RemovalRange = PrevDecl->getSourceRange();
#if 0
  SourceRange ExtendedRange = RemovalRange;

  SourceLocation ExpansionEnd = Result.SourceManager->getExpansionLoc(RemovalRange.getEnd());

  // Создать Lexer для поиска токенов после expansion конца
  SourceLocation FileStart = Result.SourceManager->getLocForStartOfFile(
      Result.SourceManager->getFileID(ExpansionEnd));
  std::pair<FileID, unsigned> LocInfo = Result.SourceManager->getDecomposedLoc(ExpansionEnd);
  StringRef FileBuffer = Result.SourceManager->getBufferData(LocInfo.first);

  Lexer Lex(ExpansionEnd, getLangOpts(), 
            FileBuffer.data(), FileBuffer.data() + LocInfo.second, 
            FileBuffer.data() + FileBuffer.size());

  Token Tok;
  if (!Lex.LexFromRawLexer(Tok) && Tok.is(tok::semi)) {
      ExtendedRange.setEnd(Result.SourceManager->getSpellingLoc(Tok.getEndLoc()));
  }
#endif
#if 0
  // Найти следующую точку с запятой после DeclStmt
  auto TokPos =
  Result.SourceManager->getExpansionLoc(RemovalRange.getBegin());
  do {
    if (const auto NextToken = Lexer::findNextToken(TokPos, 
                                                    *Result.SourceManager, 
                                                    getLangOpts())) {
        if (NextToken->is(tok::semi)) {
            ExtendedRange.setEnd(NextToken->getEndLoc());
            break;
        }
        TokPos = NextToken->getLocation();
    }
  } while (true);
#endif

  CharSourceRange DeclCharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(RemovalRange),
      *Result.SourceManager, getLangOpts());

  // Get the source text using makeFileCharRange to properly handle macros
  //const CharSourceRange DeclCharRange = Lexer::makeFileCharRange(
  //    CharSourceRange::getTokenRange(RemovalRange),
  //    *Result.SourceManager, getLangOpts());
  
  bool CanFix = true;

  if (DeclCharRange.isInvalid()) {
    DeclCharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(RemovalRange),
      *Result.SourceManager, getLangOpts());
    if (DeclCharRange.isInvalid())  
    CanFix = false;
  }
  
  const StringRef DeclStmtText = Lexer::getSourceText(
      DeclCharRange, *Result.SourceManager, getLangOpts());
  
  if (DeclStmtText.empty())
    CanFix = false;

  const auto NewInitStmt = DeclStmtText.trim().str() +
                           ("") + " ";
  // const auto NewInitStmt = normDeclStmtText(DeclStmtText).str() + "; ";

  // Allow the fix if we can extract valid source text. We use makeFileCharRange
  // to get source text that preserves macros (like MY_INT in types). The
  // rangeCanBeFixed check ensures basic validity, but we also allow the fix
  // if the range is entirely within a macro argument (which preserves macros).
  // Additionally, if we successfully extracted source text using makeFileCharRange,
  // it means we can preserve macro spelling, so we allow the fix even if
  // rangeCanBeFixed returns false due to macro expansions in the type.
  // We check that the begin location is valid and not in a system header.
  const SourceLocation DeclBegin = RemovalRange.getBegin();
  const SourceLocation SpellingLoc =
      Result.SourceManager->getSpellingLoc(DeclBegin);
  if (CanFix)
    CanFix =
        utils::rangeCanBeFixed(RemovalRange, Result.SourceManager) ||
        utils::rangeIsEntirelyWithinMacroArgument(RemovalRange,
                                                Result.SourceManager) ||
        (SpellingLoc.isValid() &&
        !Result.SourceManager->isInSystemHeader(SpellingLoc));

  auto Diag = diag(PrevDecl->getBeginLoc(),
                   "%select{multiple variable|variable %1}0 declaration "
                   "before %select{if|switch}2 statement could be moved into "
                   "%select{if|switch}2 init statement")
              << (llvm::size(PrevDecl->decls()) == 1)
              << llvm::dyn_cast<VarDecl>(*PrevDecl->decl_begin()) << !If;
  if (CanFix)
    Diag << FixItHint::CreateRemoval(RemovalRange)
         << FixItHint::CreateInsertion(Condition->getBeginLoc(), NewInitStmt);
}

} // namespace clang::tidy::modernize
