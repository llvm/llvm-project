//===--- RefactoringActionRules.h - Clang refactoring library -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_H

#include "clang/Tooling/Refactoring/RefactoringActionRule.h"
#include "clang/Tooling/Refactoring/RefactoringActionRulesInternal.h"

namespace clang {
namespace tooling {
namespace refactoring_action_rules {

/// Creates a new refactoring action rule that invokes the given function once
/// all of the requirements are satisfied. The values produced during the
/// evaluation of requirements are passed to the given function (in the order of
/// requirements).
///
/// \param RefactoringFunction the function that will perform the refactoring
/// once the requirements are satisfied. The function must return a valid
/// refactoring result type wrapped in an \c Expected type. The following result
/// types are currently supported:
///
///  - AtomicChanges: the refactoring function will be used to create source
///                   replacements.
///
/// \param Requirements a set of rule requirements that have to be satisfied.
/// Each requirement must be a valid requirement, i.e. the value of
/// \c traits::IsRequirement<T> must be true. The following requirements are
/// currently supported:
///
///  - requiredSelection: The refactoring function won't be invoked unless the
///                       given selection requirement is satisfied.
template <typename ResultType, typename... RequirementTypes>
std::unique_ptr<RefactoringActionRule>
createRefactoringRule(Expected<ResultType> (*RefactoringFunction)(
                          typename RequirementTypes::OutputType...),
                      const RequirementTypes &... Requirements) {
  static_assert(tooling::traits::IsValidRefactoringResult<ResultType>::value,
                "invalid refactoring result type");
  static_assert(traits::IsRequirement<RequirementTypes...>::value,
                "invalid refactoring action rule requirement");
  return llvm::make_unique<internal::PlainFunctionRule<
      decltype(RefactoringFunction), RequirementTypes...>>(
      RefactoringFunction, std::make_tuple(Requirements...));
}

template <
    typename Callable, typename... RequirementTypes,
    typename Fn = decltype(&Callable::operator()),
    typename ResultType = typename internal::LambdaDeducer<Fn>::ReturnType,
    bool IsNonCapturingLambda = std::is_convertible<
        Callable,
        ResultType (*)(typename RequirementTypes::OutputType...)>::value,
    typename = typename std::enable_if<IsNonCapturingLambda>::type>
std::unique_ptr<RefactoringActionRule>
createRefactoringRule(const Callable &C,
                      const RequirementTypes &... Requirements) {
  ResultType (*Func)(typename RequirementTypes::OutputType...) = C;
  return createRefactoringRule(Func, Requirements...);
}

} // end namespace refactoring_action_rules
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULES_H
