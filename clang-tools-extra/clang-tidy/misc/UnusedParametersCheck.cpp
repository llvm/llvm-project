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
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include <unordered_set>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {
bool isOverrideMethod(const FunctionDecl *Function) {
  if (const auto *MD = dyn_cast<CXXMethodDecl>(Function))
    return MD->size_overridden_methods() > 0 || MD->hasAttr<OverrideAttr>();
  return false;
}
} // namespace

void UnusedParametersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), hasBody(stmt()), hasAnyParameter(decl()))
          .bind("function"),
      this);
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

class UnusedParametersCheck::IndexerVisitor
    : public RecursiveASTVisitor<IndexerVisitor> {
public:
  IndexerVisitor(TranslationUnitDecl *Top) { TraverseDecl(Top); }

  const std::unordered_set<const CallExpr *> &
  getFnCalls(const FunctionDecl *Fn) {
    return Index[Fn->getCanonicalDecl()].Calls;
  }

  const std::unordered_set<const DeclRefExpr *> &
  getOtherRefs(const FunctionDecl *Fn) {
    return Index[Fn->getCanonicalDecl()].OtherRefs;
  }

  bool shouldTraversePostOrder() const { return true; }

  bool WalkUpFromDeclRefExpr(DeclRefExpr *DeclRef) {
    if (const auto *Fn = dyn_cast<FunctionDecl>(DeclRef->getDecl())) {
      Fn = Fn->getCanonicalDecl();
      Index[Fn].OtherRefs.insert(DeclRef);
    }
    return true;
  }

  bool WalkUpFromCallExpr(CallExpr *Call) {
    if (const auto *Fn =
            dyn_cast_or_null<FunctionDecl>(Call->getCalleeDecl())) {
      Fn = Fn->getCanonicalDecl();
      if (const auto *Ref =
              dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImplicit())) {
        Index[Fn].OtherRefs.erase(Ref);
      }
      Index[Fn].Calls.insert(Call);
    }
    return true;
  }

private:
  struct IndexEntry {
    std::unordered_set<const CallExpr *> Calls;
    std::unordered_set<const DeclRefExpr *> OtherRefs;
  };

  std::unordered_map<const FunctionDecl *, IndexEntry> Index;
};

UnusedParametersCheck::~UnusedParametersCheck() = default;

UnusedParametersCheck::UnusedParametersCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", 0) != 0) {}

void UnusedParametersCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void UnusedParametersCheck::warnOnUnusedParameter(
    const MatchFinder::MatchResult &Result, const FunctionDecl *Function,
    unsigned ParamIndex) {
  const auto *Param = Function->getParamDecl(ParamIndex);
  auto MyDiag = diag(Param->getLocation(), "parameter %0 is unused") << Param;

  if (!Indexer) {
    Indexer = llvm::make_unique<IndexerVisitor>(
        Result.Context->getTranslationUnitDecl());
  }

  // Comment out parameter name for non-local functions.
  if (Function->isExternallyVisible() ||
      !Result.SourceManager->isInMainFile(Function->getLocation()) ||
      !Indexer->getOtherRefs(Function).empty() || isOverrideMethod(Function)) {
    SourceRange RemovalRange(Param->getLocation());
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
  for (const auto *Call : Indexer->getFnCalls(Function))
    MyDiag << removeArgument(Result, Call, ParamIndex);
}

void UnusedParametersCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function->hasWrittenPrototype() || Function->isTemplateInstantiation())
    return;
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Function))
    if (Method->isLambdaStaticInvoker())
      return;
  for (unsigned i = 0, e = Function->getNumParams(); i != e; ++i) {
    const auto *Param = Function->getParamDecl(i);
    if (Param->isUsed() || Param->isReferenced() || !Param->getDeclName() ||
        Param->hasAttr<UnusedAttr>())
      continue;

    // In non-strict mode ignore function definitions with empty bodies
    // (constructor initializer counts for non-empty body).
    if (StrictMode ||
        (Function->getBody()->child_begin() !=
         Function->getBody()->child_end()) ||
        (isa<CXXConstructorDecl>(Function) &&
         cast<CXXConstructorDecl>(Function)->getNumCtorInitializers() > 0))
      warnOnUnusedParameter(Result, Function, i);
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
