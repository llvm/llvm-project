//===--- UnusedParametersCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnusedParametersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

void UnusedParametersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl().bind("function"), this);
}

template <typename T>
static CharSourceRange removeNode(const MatchFinder::MatchResult &Result,
                                  const T *PrevNode, const T *Node,
                                  const T *NextNode) {
  if (NextNode)
    return CharSourceRange::getCharRange(Node->getLocStart(),
                                         NextNode->getLocStart());

  if (PrevNode)
    return CharSourceRange::getTokenRange(
        Lexer::getLocForEndOfToken(PrevNode->getLocEnd(), 0,
                                   *Result.SourceManager,
                                   Result.Context->getLangOpts()),
        Node->getLocEnd());

  return CharSourceRange::getTokenRange(Node->getSourceRange());
}

static FixItHint removeParameter(const MatchFinder::MatchResult &Result,
                                 const FunctionDecl *Function, unsigned Index) {
  return FixItHint::CreateRemoval(removeNode(
      Result, Index > 0 ? Function->getParamDecl(Index - 1) : nullptr,
      Function->getParamDecl(Index),
      Index + 1 < Function->getNumParams() ? Function->getParamDecl(Index + 1)
                                           : nullptr));
}

static FixItHint removeArgument(const MatchFinder::MatchResult &Result,
                                const CallExpr *Call, unsigned Index) {
  return FixItHint::CreateRemoval(removeNode(
      Result, Index > 0 ? Call->getArg(Index - 1) : nullptr,
      Call->getArg(Index),
      Index + 1 < Call->getNumArgs() ? Call->getArg(Index + 1) : nullptr));
}

void UnusedParametersCheck::warnOnUnusedParameter(
    const MatchFinder::MatchResult &Result, const FunctionDecl *Function,
    unsigned ParamIndex) {
  const auto *Param = Function->getParamDecl(ParamIndex);
  auto MyDiag = diag(Param->getLocation(), "parameter '%0' is unused")
                << Param->getName();

  auto UsedByRef = [&] {
    return !ast_matchers::match(
                decl(hasDescendant(
                    declRefExpr(to(equalsNode(Function)),
                                unless(hasAncestor(
                                    callExpr(callee(equalsNode(Function)))))))),
                *Result.Context->getTranslationUnitDecl(), *Result.Context)
                .empty();
  };

  // Comment out parameter name for non-local functions.
  if (Function->isExternallyVisible() ||
      !Result.SourceManager->isInMainFile(Function->getLocation()) ||
      UsedByRef()) {
    SourceRange RemovalRange(Param->getLocation(), Param->getLocEnd());
    // Note: We always add a space before the '/*' to not accidentally create a
    // '*/*' for pointer types, which doesn't start a comment. clang-format will
    // clean this up afterwards.
    MyDiag << FixItHint::CreateReplacement(
        RemovalRange, (Twine(" /*") + Param->getName() + "*/").str());
    return;
  }

  // Fix all redeclarations.
  for (const FunctionDecl *FD : Function->redecls())
    if (FD->param_size())
      MyDiag << removeParameter(Result, FD, ParamIndex);

  // Fix all call sites.
  auto CallMatches = ast_matchers::match(
      decl(forEachDescendant(
          callExpr(callee(functionDecl(equalsNode(Function)))).bind("x"))),
      *Result.Context->getTranslationUnitDecl(), *Result.Context);
  for (const auto &Match : CallMatches)
    MyDiag << removeArgument(Result, Match.getNodeAs<CallExpr>("x"),
                             ParamIndex);
}

void UnusedParametersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function->doesThisDeclarationHaveABody() ||
      !Function->hasWrittenPrototype())
    return;
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Function))
    if (Method->isLambdaStaticInvoker())
      return;
  for (unsigned i = 0, e = Function->getNumParams(); i != e; ++i) {
    const auto *Param = Function->getParamDecl(i);
    if (Param->isUsed() || Param->isReferenced() || !Param->getDeclName() ||
        Param->hasAttr<UnusedAttr>())
      continue;
    warnOnUnusedParameter(Result, Function, i);
  }
}

} // namespace tidy
} // namespace clang
