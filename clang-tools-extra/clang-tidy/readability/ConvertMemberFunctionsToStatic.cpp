//===--- ConvertMemberFunctionsToStatic.cpp - clang-tidy ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ConvertMemberFunctionsToStatic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

AST_MATCHER(CXXMethodDecl, isStatic) { return Node.isStatic(); }

AST_MATCHER(CXXMethodDecl, hasTrivialBody) { return Node.hasTrivialBody(); }

AST_MATCHER(CXXMethodDecl, isOverloadedOperator) {
  return Node.isOverloadedOperator();
}

AST_MATCHER(CXXRecordDecl, hasAnyDependentBases) {
  return Node.hasAnyDependentBases();
}

AST_MATCHER(CXXMethodDecl, isTemplate) {
  return Node.getTemplatedKind() != FunctionDecl::TK_NonTemplate;
}

AST_MATCHER(CXXMethodDecl, isDependentContext) {
  return Node.isDependentContext();
}

AST_MATCHER(CXXMethodDecl, isInsideMacroDefinition) {
  const ASTContext &Ctxt = Finder->getASTContext();
  return clang::Lexer::makeFileCharRange(
             clang::CharSourceRange::getCharRange(
                 Node.getTypeSourceInfo()->getTypeLoc().getSourceRange()),
             Ctxt.getSourceManager(), Ctxt.getLangOpts())
      .isInvalid();
}

AST_MATCHER_P(CXXMethodDecl, hasCanonicalDecl,
              ast_matchers::internal::Matcher<CXXMethodDecl>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getCanonicalDecl(), Finder, Builder);
}

AST_MATCHER(CXXMethodDecl, usesThis) {
  class FindUsageOfThis : public RecursiveASTVisitor<FindUsageOfThis> {
  public:
    bool Used = false;

    bool VisitCXXThisExpr(const CXXThisExpr *E) {
      Used = true;
      return false; // Stop traversal.
    }
  } UsageOfThis;

  // TraverseStmt does not modify its argument.
  UsageOfThis.TraverseStmt(const_cast<Stmt *>(Node.getBody()));

  return UsageOfThis.Used;
}

void ConvertMemberFunctionsToStatic::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(
          isDefinition(), isUserProvided(),
          unless(anyOf(
              isExpansionInSystemHeader(), isVirtual(), isStatic(),
              hasTrivialBody(), isOverloadedOperator(), cxxConstructorDecl(),
              cxxDestructorDecl(), cxxConversionDecl(), isTemplate(),
              isDependentContext(),
              ofClass(anyOf(
                  isLambda(),
                  hasAnyDependentBases()) // Method might become virtual
                                          // depending on template base class.
                      ),
              isInsideMacroDefinition(),
              hasCanonicalDecl(isInsideMacroDefinition()), usesThis())))
          .bind("x"),
      this);
}

/// Obtain the original source code text from a SourceRange.
static StringRef getStringFromRange(SourceManager &SourceMgr,
                                    const LangOptions &LangOpts,
                                    SourceRange Range) {
  if (SourceMgr.getFileID(Range.getBegin()) !=
      SourceMgr.getFileID(Range.getEnd()))
    return {};

  return Lexer::getSourceText(CharSourceRange(Range, true), SourceMgr,
                              LangOpts);
}

static SourceRange getLocationOfConst(const TypeSourceInfo *TSI,
                                      SourceManager &SourceMgr,
                                      const LangOptions &LangOpts) {
  assert(TSI);
  const auto FTL = TSI->getTypeLoc().IgnoreParens().getAs<FunctionTypeLoc>();
  assert(FTL);

  SourceRange Range{FTL.getRParenLoc().getLocWithOffset(1),
                    FTL.getLocalRangeEnd()};
  // Inside Range, there might be other keywords and trailing return types.
  // Find the exact position of "const".
  StringRef Text = getStringFromRange(SourceMgr, LangOpts, Range);
  size_t Offset = Text.find("const");
  if (Offset == StringRef::npos)
    return {};

  SourceLocation Start = Range.getBegin().getLocWithOffset(Offset);
  return {Start, Start.getLocWithOffset(strlen("const") - 1)};
}

void ConvertMemberFunctionsToStatic::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Definition = Result.Nodes.getNodeAs<CXXMethodDecl>("x");

  // TODO: For out-of-line declarations, don't modify the source if the header
  // is excluded by the -header-filter option.
  DiagnosticBuilder Diag =
      diag(Definition->getLocation(), "method %0 can be made static")
      << Definition;

  // TODO: Would need to remove those in a fix-it.
  if (Definition->getMethodQualifiers().hasVolatile() ||
      Definition->getMethodQualifiers().hasRestrict() ||
      Definition->getRefQualifier() != RQ_None)
    return;

  const CXXMethodDecl *Declaration = Definition->getCanonicalDecl();

  if (Definition->isConst()) {
    // Make sure that we either remove 'const' on both declaration and
    // definition or emit no fix-it at all.
    SourceRange DefConst = getLocationOfConst(Definition->getTypeSourceInfo(),
                                              *Result.SourceManager,
                                              Result.Context->getLangOpts());

    if (DefConst.isInvalid())
      return;

    if (Declaration != Definition) {
      SourceRange DeclConst = getLocationOfConst(
          Declaration->getTypeSourceInfo(), *Result.SourceManager,
          Result.Context->getLangOpts());

      if (DeclConst.isInvalid())
        return;
      Diag << FixItHint::CreateRemoval(DeclConst);
    }

    // Remove existing 'const' from both declaration and definition.
    Diag << FixItHint::CreateRemoval(DefConst);
  }
  Diag << FixItHint::CreateInsertion(Declaration->getBeginLoc(), "static ");
}

} // namespace readability
} // namespace tidy
} // namespace clang
