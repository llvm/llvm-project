//===--- RefactoringActionRuleRequirements.h - Clang refactoring library --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_REQUIREMENTS_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_REQUIREMENTS_H

#include "clang/Tooling/Refactoring/RefactoringActionRuleRequirementsInternal.h"
#include "llvm/Support/Error.h"
#include <type_traits>

namespace clang {
namespace tooling {
namespace refactoring_action_rules {

/// Creates a selection requirement from the given requirement.
///
/// Requirements must subclass \c selection::Requirement and implement
/// evaluateSelection member function.
template <typename T>
internal::SourceSelectionRequirement<
    typename selection::internal::EvaluateSelectionChecker<
        decltype(&T::evaluateSelection)>::ArgType,
    typename selection::internal::EvaluateSelectionChecker<
        decltype(&T::evaluateSelection)>::ReturnType,
    T>
requiredSelection(
    const T &Requirement,
    typename std::enable_if<selection::traits::IsRequirement<T>::value>::type
        * = nullptr) {
  return internal::SourceSelectionRequirement<
      typename selection::internal::EvaluateSelectionChecker<decltype(
          &T::evaluateSelection)>::ArgType,
      typename selection::internal::EvaluateSelectionChecker<decltype(
          &T::evaluateSelection)>::ReturnType,
      T>(Requirement);
}

template <typename T>
void requiredSelection(
    const T &,
    typename std::enable_if<
        !std::is_base_of<selection::Requirement, T>::value>::type * = nullptr) {
  static_assert(
      sizeof(T) && false,
      "selection requirement must be a class derived from Requirement");
}

} // end namespace refactoring_action_rules
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_REQUIREMENTS_H
