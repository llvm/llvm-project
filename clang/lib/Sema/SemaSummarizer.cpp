#include "clang/Sema/SemaSummarizer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/SummaryConsumer.h"
#include <set>

namespace clang {
namespace {
class CallCollector : public ast_matchers::MatchFinder::MatchCallback {
  FunctionSummary *Summary;

  CallCollector(FunctionSummary &Summary) : Summary(&Summary) {}

  virtual void
  run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
    if (!Call)
      return;

    const auto *Callee = llvm::dyn_cast<FunctionDecl>(Call->getCalleeDecl());
    Summary->addCall(Callee);
  }

public:
  static void CollectCalledFunctions(const FunctionDecl *FD,
                                     FunctionSummary &Summary) {
    using namespace ast_matchers;
    MatchFinder Finder;
    CallCollector CC(Summary);

    Finder.addMatcher(functionDecl(forEachDescendant(callExpr().bind("call"))),
                      &CC);
    Finder.match(*FD, FD->getASTContext());
  }
};

class NoWriteGlobalAttrManager : public SummaryAttributeManager {
  class Callback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool WriteGlobal = false;

    void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
      const auto *Assignment =
          Result.Nodes.getNodeAs<BinaryOperator>("assignment");
      if (!Assignment)
        return;

      WriteGlobal = true;
    };
  };

public:
  NoWriteGlobalAttrManager()
      : SummaryAttributeManager(NO_WRITE_GLOBAL, "no_write_global") {}

  bool predicate(const FunctionDecl *FD) override {
    using namespace ast_matchers;
    MatchFinder Finder;
    Callback CB;

    Finder.addMatcher(
        functionDecl(forEachDescendant(
            binaryOperator(isAssignmentOperator(),
                           hasLHS(declRefExpr(to(varDecl(hasGlobalStorage())))))
                .bind("assignment"))),
        &CB);
    Finder.match(*FD, FD->getASTContext());
    return !CB.WriteGlobal;
  };

  bool merge(FunctionSummary &Summary) override { return true; };
};
} // namespace

void FunctionSummary::addCall(const clang::FunctionDecl *FD) {
  SmallVector<char> Call;
  index::generateUSRForDecl(FD, Call);
  Calls.emplace(Call);
}

FunctionSummary::FunctionSummary(const clang::FunctionDecl *FD) {
  index::generateUSRForDecl(FD, ID);
}

SemaSummarizer::SemaSummarizer(Sema &S, SummaryConsumer *SummaryConsumer)
    : SemaBase(S), TheSummaryConsumer(SummaryConsumer) {
  Attributes.emplace_back(std::make_unique<NoWriteGlobalAttrManager>());
}

void SemaSummarizer::ActOnStartOfSourceFile() {
  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessStartOfSourceFile();
}

void SemaSummarizer::ActOnEndOfSourceFile() {
  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessEndOfSourceFile();
}

void SemaSummarizer::SummarizeFunctionBody(const FunctionDecl *FD) {
  FunctionSummary Summary(FD);
  CallCollector::CollectCalledFunctions(FD, Summary);

  for (auto &&Attr : Attributes) {
    if (const auto &InferredAttr = Attr->infer(FD))
      Summary.addAttribute(*InferredAttr);
  }

  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessFunctionSummary(Summary);
}

} // namespace clang
