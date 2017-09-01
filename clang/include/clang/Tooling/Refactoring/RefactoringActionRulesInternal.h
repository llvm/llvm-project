//===--- RefactoringActionRulesInternal.h - Clang refactoring library -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_INTERNAL_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_INTERNAL_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactoring/RefactoringActionRule.h"
#include "clang/Tooling/Refactoring/RefactoringActionRuleRequirements.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"
#include "llvm/Support/Error.h"
#include <type_traits>

namespace clang {
namespace tooling {
namespace refactoring_action_rules {
namespace internal {

/// A specialized refactoring action rule that calls the stored function once
/// all the of the requirements are fullfilled. The values produced during the
/// evaluation of requirements are passed to the stored function.
template <typename FunctionType, typename... RequirementTypes>
class PlainFunctionRule final : public RefactoringActionRule {
public:
  PlainFunctionRule(FunctionType Function,
                    std::tuple<RequirementTypes...> &&Requirements)
      : Function(Function), Requirements(std::move(Requirements)) {}

  void invoke(RefactoringResultConsumer &Consumer,
              RefactoringRuleContext &Context) override {
    return invokeImpl(Consumer, Context,
                      llvm::index_sequence_for<RequirementTypes...>());
  }

private:
  /// Returns \c T when given \c Expected<Optional<T>>, or \c T otherwise.
  template <typename T>
  static T &&unwrapRequirementResult(Expected<Optional<T>> &&X) {
    assert(X && "unexpected diagnostic!");
    return std::move(**X);
  }
  template <typename T> static T &&unwrapRequirementResult(T &&X) {
    return std::move(X);
  }

  /// Scans the tuple and returns a \c PartialDiagnosticAt
  /// from the first invalid \c DiagnosticOr value. Returns \c None if all
  /// values are valid.
  template <typename FirstT, typename... RestT>
  static Optional<llvm::Error> findErrorNone(FirstT &First, RestT &... Rest) {
    Optional<llvm::Error> Result = takeErrorNone(First);
    if (Result)
      return Result;
    return findErrorNone(Rest...);
  }

  static Optional<llvm::Error> findErrorNone() { return None; }

  template <typename T> static Optional<llvm::Error> takeErrorNone(T &) {
    return None;
  }

  template <typename T>
  static Optional<llvm::Error> takeErrorNone(Expected<Optional<T>> &Diag) {
    if (!Diag)
      return std::move(Diag.takeError());
    if (!*Diag)
      return llvm::Error::success(); // Initiation failed without a diagnostic.
    return None;
  }

  template <size_t... Is>
  void invokeImpl(RefactoringResultConsumer &Consumer,
                  RefactoringRuleContext &Context,
                  llvm::index_sequence<Is...>) {
    // Initiate the operation.
    auto Values =
        std::make_tuple(std::get<Is>(Requirements).evaluate(Context)...);
    Optional<llvm::Error> InitiationFailure =
        findErrorNone(std::get<Is>(Values)...);
    if (InitiationFailure) {
      llvm::Error Error = std::move(*InitiationFailure);
      if (!Error)
        // FIXME: Use a diagnostic.
        return Consumer.handleError(llvm::make_error<llvm::StringError>(
            "refactoring action can't be initiated with the specified "
            "selection range",
            llvm::inconvertibleErrorCode()));
      return Consumer.handleError(std::move(Error));
    }
    // Perform the operation.
    auto Result =
        Function(unwrapRequirementResult(std::move(std::get<Is>(Values)))...);
    if (!Result)
      return Consumer.handleError(Result.takeError());
    Consumer.handle(std::move(*Result));
  }

  FunctionType Function;
  std::tuple<RequirementTypes...> Requirements;
};

/// Used to deduce the refactoring result type for the lambda that passed into
/// createRefactoringRule.
template <typename T> struct LambdaDeducer;
template <typename T, typename R, typename... Args>
struct LambdaDeducer<R (T::*)(Args...) const> {
  using ReturnType = R;
};

} // end namespace internal
} // end namespace refactoring_action_rules
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_INTERNAL_H
