#include "clang/Summary/SummaryAttribute.h"
#include "clang/Summary/SummaryContext.h"

namespace clang {
bool NoWriteGlobalAttr::infer(const FunctionDecl *FD) const {
  using namespace ast_matchers;
  MatchFinder Finder;

  class Callback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool WriteGlobal = false;

    void
    run(const ast_matchers::MatchFinder::MatchResult &Result) override final {
      const auto *Assignment =
          Result.Nodes.getNodeAs<BinaryOperator>("assignment");
      if (!Assignment)
        return;

      WriteGlobal = true;
    }
  } CB;

  Finder.addMatcher(
      functionDecl(forEachDescendant(
          binaryOperator(isAssignmentOperator(),
                         hasLHS(declRefExpr(to(varDecl(hasGlobalStorage())))))
              .bind("assignment"))),
      &CB);
  Finder.match(*FD, FD->getASTContext());
  return !CB.WriteGlobal;
}

bool NoWriteGlobalAttr::merge(const FunctionSummary &Caller,
                              const FunctionSummary *Callee) const {
  return !Caller.callsOpaqueObject() && Caller.getAttributes().count(this) &&
         Callee && Callee->getAttributes().count(this);
}
} // namespace clang
