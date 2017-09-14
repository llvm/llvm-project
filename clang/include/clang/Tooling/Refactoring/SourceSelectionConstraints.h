//===--- SourceSelectionConstraints.h - Clang refactoring library ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_SOURCE_SELECTION_CONSTRAINTS_H
#define LLVM_CLANG_TOOLING_REFACTOR_SOURCE_SELECTION_CONSTRAINTS_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"
#include <type_traits>

namespace clang {
namespace tooling {

class RefactoringRuleContext;

namespace selection {

/// This constraint is satisfied when any portion of the source text is
/// selected. It can be used to obtain the raw source selection range.
struct SourceSelectionRange {
  SourceSelectionRange(const SourceManager &SM, SourceRange Range)
      : SM(SM), Range(Range) {}

  const SourceManager &getSources() const { return SM; }
  SourceRange getRange() const { return Range; }

  static Optional<SourceSelectionRange>
  evaluate(RefactoringRuleContext &Context);

private:
  const SourceManager &SM;
  SourceRange Range;
};

/// A custom selection requirement.
class Requirement {
  /// Subclasses must implement
  /// 'T evaluateSelection(const RefactoringRuleContext &,
  /// SelectionConstraint) const' member function. \c T is used to determine
  /// the return type that is passed to the refactoring rule's function.
  /// If T is \c DiagnosticOr<S> , then \c S is passed to the rule's function
  /// using move semantics.
  /// Otherwise, T is passed to the function directly using move semantics.
  ///
  /// The different return type rules allow refactoring actions to fail
  /// initiation when the relevant portions of AST aren't selected.
};

namespace traits {

/// A type trait that returns true iff the given type is a valid selection
/// constraint.
template <typename T> struct IsConstraint : public std::false_type {};

} // end namespace traits

namespace internal {

template <typename T> struct EvaluateSelectionChecker : std::false_type {};

template <typename T, typename R, typename A>
struct EvaluateSelectionChecker<R (T::*)(const RefactoringRuleContext &, A)
                                    const> : std::true_type {
  using ReturnType = R;
  using ArgType = A;
};

template <typename T> class Identity : public Requirement {
public:
  T evaluateSelection(const RefactoringRuleContext &, T Value) const {
    return std::move(Value);
  }
};

} // end namespace internal

/// A identity function that returns the given selection constraint is provided
/// for convenience, as it can be passed to \c requiredSelection directly.
template <typename T> internal::Identity<T> identity() {
  static_assert(
      traits::IsConstraint<T>::value,
      "selection::identity can be used with selection constraints only");
  return internal::Identity<T>();
}

namespace traits {

template <>
struct IsConstraint<SourceSelectionRange> : public std::true_type {};

/// A type trait that returns true iff \c T is a valid selection requirement.
template <typename T>
struct IsRequirement
    : std::conditional<
          std::is_base_of<Requirement, T>::value &&
              internal::EvaluateSelectionChecker<decltype(
                  &T::evaluateSelection)>::value &&
              IsConstraint<typename internal::EvaluateSelectionChecker<decltype(
                  &T::evaluateSelection)>::ArgType>::value,
          std::true_type, std::false_type>::type {};

} // end namespace traits
} // end namespace selection
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_SOURCE_SELECTION_CONSTRAINTS_H
