//===--- RefactoringActionRuleRequirementsInternal.h - --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_REQUIREMENTS_INTERNAL_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_REQUIREMENTS_INTERNAL_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"
#include "clang/Tooling/Refactoring/SourceSelectionConstraints.h"
#include <type_traits>

namespace clang {
namespace tooling {
namespace refactoring_action_rules {
namespace internal {

/// A base class for any requirement. Used by the \c IsRequirement trait to
/// determine if a class is a valid requirement.
struct RequirementBase {};

/// Defines a type alias of type \T when given \c Expected<Optional<T>>, or
/// \c T otherwise.
template <typename T> struct DropExpectedOptional { using Type = T; };

template <typename T> struct DropExpectedOptional<Expected<Optional<T>>> {
  using Type = T;
};

/// The \c requiredSelection refactoring action requirement is represented
/// using this type.
template <typename InputT, typename OutputT, typename RequirementT>
struct SourceSelectionRequirement
    : std::enable_if<selection::traits::IsConstraint<InputT>::value &&
                         selection::traits::IsRequirement<RequirementT>::value,
                     RequirementBase>::type {
  using OutputType = typename DropExpectedOptional<OutputT>::Type;

  SourceSelectionRequirement(const RequirementT &Requirement)
      : Requirement(Requirement) {}

  /// Evaluates the action rule requirement by ensuring that both the selection
  /// constraint and the selection requirement can be evaluated with the given
  /// context.
  ///
  /// \returns None if the selection constraint is not evaluated successfully,
  /// Error if the selection requirement is not evaluated successfully or
  /// an OutputT if the selection requirement was successfully. The OutpuT
  /// value is wrapped in Expected<Optional<>> which is then unwrapped by the
  /// refactoring action rule before passing the value to the refactoring
  /// function.
  Expected<Optional<OutputType>> evaluate(RefactoringRuleContext &Context) {
    Optional<InputT> Value = InputT::evaluate(Context);
    if (!Value)
      return None;
    return std::move(Requirement.evaluateSelection(Context, *Value));
  }

private:
  const RequirementT Requirement;
};

} // end namespace internal

namespace traits {

/// A type trait that returns true iff the given type is a valid rule
/// requirement.
template <typename First, typename... Rest>
struct IsRequirement : std::conditional<IsRequirement<First>::value &&
                                            IsRequirement<Rest...>::value,
                                        std::true_type, std::false_type>::type {
};

template <typename T>
struct IsRequirement<T>
    : std::conditional<std::is_base_of<internal::RequirementBase, T>::value,
                       std::true_type, std::false_type>::type {};

/// A type trait that returns true when the given type has at least one source
/// selection requirement.
template <typename First, typename... Rest>
struct HasSelectionRequirement
    : std::conditional<HasSelectionRequirement<First>::value ||
                           HasSelectionRequirement<Rest...>::value,
                       std::true_type, std::false_type>::type {};

template <typename I, typename O, typename R>
struct HasSelectionRequirement<internal::SourceSelectionRequirement<I, O, R>>
    : std::true_type {};

template <typename T> struct HasSelectionRequirement<T> : std::false_type {};

} // end namespace traits
} // end namespace refactoring_action_rules
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_REQUIREMENTS_INTERNAL_H
