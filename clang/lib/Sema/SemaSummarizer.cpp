#include "clang/Sema/SemaSummarizer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Index/USRGeneration.h"
#include <set>

namespace clang {
namespace {
class FunctionSummary {
  SmallVector<char> ID;
  std::vector<std::string> FunctionAttrs;
  std::set<SmallVector<char>> Calls;

public:
  void addCall(const clang::FunctionDecl *FD) {
    SmallVector<char> Call;
    index::generateUSRForDecl(FD, Call);
    Calls.emplace(Call);
  }

  FunctionSummary(const clang::FunctionDecl *FD) {
    index::generateUSRForDecl(FD, ID);
  }
};

class CallCollector : public ast_matchers::MatchFinder::MatchCallback {
  FunctionSummary *Summary;

public:
  CallCollector(FunctionSummary &Summary) : Summary(&Summary) {}

  virtual void
  run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
    if (!Call)
      return;

    const auto *Callee = llvm::dyn_cast<FunctionDecl>(Call->getCalleeDecl());
    Summary->addCall(Callee);
  }
};

void CollectCalledFunctions(const FunctionDecl *FD, FunctionSummary &Summary) {
  using namespace ast_matchers;
  MatchFinder Finder;
  CallCollector CC(Summary);

  Finder.addMatcher(functionDecl(forEachDescendant(callExpr().bind("call"))),
                    &CC);
  Finder.match(*FD, FD->getASTContext());
}

} // namespace

void SemaSummarizer::SummarizeFunctionBody(const FunctionDecl *FD) {
  FunctionSummary Summary(FD);
  CollectCalledFunctions(FD, Summary);
}

} // namespace clang
