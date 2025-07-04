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

bool NoWritePtrParameterAttr::infer(const FunctionDecl *FD) const {
  using namespace ast_matchers;
  MatchFinder Finder;

  class Callback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool MayWritePtrParam = false;

    void
    run(const ast_matchers::MatchFinder::MatchResult &Result) override final {
      const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("fn");
      if (!FD)
        return;

      MayWritePtrParam = true;
    }
  } CB;

  auto ptrParmDeclRef = declRefExpr(
      allOf(unless(hasAncestor(unaryOperator(hasOperatorName("*")))),
            to(parmVarDecl(hasType(pointerType())))));
  auto ptrParmDereference = unaryOperator(allOf(
      hasOperatorName("*"),
      hasDescendant(declRefExpr(to(parmVarDecl(hasType(pointerType())))))));

  Finder.addMatcher(
      functionDecl(
          anyOf(
              // The value of the pointer is used to initialize a local
              // variable.
              forEachDescendant(
                  varDecl(hasInitializer(hasDescendant(ptrParmDeclRef)))),
              // The ptr parameter appears on the RHS of an assignment.
              forEachDescendant(
                  binaryOperator(isAssignmentOperator(),
                                 hasRHS(hasDescendant(ptrParmDeclRef)))),
              // The ptr is dereferenced on the LHS of an assignment.
              forEachDescendant(binaryOperator(
                  isAssignmentOperator(),
                  hasLHS(anyOf(ptrParmDereference,
                               hasDescendant(ptrParmDereference))))),
              // The param is const casted
              forEachDescendant(cxxConstCastExpr(hasDescendant(ptrParmDeclRef)))
              // FIXME: handle member access
              ))
          .bind("fn"),
      &CB);
  Finder.match(*FD, FD->getASTContext());
  return !CB.MayWritePtrParam;
}

bool NoWritePtrParameterAttr::merge(const FunctionSummary &Caller,
                                    const FunctionSummary *Callee) const {
  return !Caller.callsOpaqueObject() && Caller.getAttributes().count(this) &&
         Callee && Callee->getAttributes().count(this);
}
} // namespace clang
