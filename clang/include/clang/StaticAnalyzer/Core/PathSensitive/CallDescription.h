//===- CallDescription.h - function/method call matching       --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines a generic mechanism for matching for function and
/// method calls of C, C++, and Objective-C languages. Instances of these
/// classes are frequently used together with the CallEvent classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CALLDESCRIPTION_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CALLDESCRIPTION_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include <vector>

namespace clang {
class IdentifierInfo;
} // namespace clang

namespace clang {
namespace ento {

enum CallDescriptionFlags : unsigned {
  CDF_None = 0,

  /// Describes a C standard function that is sometimes implemented as a macro
  /// that expands to a compiler builtin with some __builtin prefix.
  /// The builtin may as well have a few extra arguments on top of the requested
  /// number of arguments.
  CDF_MaybeBuiltin = 1 << 0,
};

/// This class represents a description of a function call using the number of
/// arguments and the name of the function.
class CallDescription {
  friend class CallEvent;
  using MaybeCount = Optional<unsigned>;

  mutable Optional<const IdentifierInfo *> II;
  // The list of the qualified names used to identify the specified CallEvent,
  // e.g. "{a, b}" represent the qualified names, like "a::b".
  std::vector<std::string> QualifiedName;
  MaybeCount RequiredArgs;
  MaybeCount RequiredParams;
  int Flags;

public:
  /// Constructs a CallDescription object.
  ///
  /// @param QualifiedName The list of the name qualifiers of the function that
  /// will be matched. The user is allowed to skip any of the qualifiers.
  /// For example, {"std", "basic_string", "c_str"} would match both
  /// std::basic_string<...>::c_str() and std::__1::basic_string<...>::c_str().
  ///
  /// @param RequiredArgs The number of arguments that is expected to match a
  /// call. Omit this parameter to match every occurrence of call with a given
  /// name regardless the number of arguments.
  CallDescription(CallDescriptionFlags Flags,
                  ArrayRef<const char *> QualifiedName,
                  MaybeCount RequiredArgs = None,
                  MaybeCount RequiredParams = None);

  /// Construct a CallDescription with default flags.
  CallDescription(ArrayRef<const char *> QualifiedName,
                  MaybeCount RequiredArgs = None,
                  MaybeCount RequiredParams = None);

  CallDescription(std::nullptr_t) = delete;

  /// Get the name of the function that this object matches.
  StringRef getFunctionName() const { return QualifiedName.back(); }

  /// Get the qualified name parts in reversed order.
  /// E.g. { "std", "vector", "data" } -> "vector", "std"
  auto begin_qualified_name_parts() const {
    return std::next(QualifiedName.rbegin());
  }
  auto end_qualified_name_parts() const { return QualifiedName.rend(); }

  /// It's false, if and only if we expect a single identifier, such as
  /// `getenv`. It's true for `std::swap`, or `my::detail::container::data`.
  bool hasQualifiedNameParts() const { return QualifiedName.size() > 1; }

  /// @name Matching CallDescriptions against a CallEvent
  /// @{

  /// Returns true if the CallEvent is a call to a function that matches
  /// the CallDescription.
  ///
  /// \note This function is not intended to be used to match Obj-C method
  /// calls.
  bool matches(const CallEvent &Call) const;

  /// Returns true whether the CallEvent matches on any of the CallDescriptions
  /// supplied.
  ///
  /// \note This function is not intended to be used to match Obj-C method
  /// calls.
  friend bool matchesAny(const CallEvent &Call, const CallDescription &CD1) {
    return CD1.matches(Call);
  }

  /// \copydoc clang::ento::matchesAny(const CallEvent &, const CallDescription &)
  template <typename... Ts>
  friend bool matchesAny(const CallEvent &Call, const CallDescription &CD1,
                         const Ts &...CDs) {
    return CD1.matches(Call) || matchesAny(Call, CDs...);
  }
  /// @}
};

/// An immutable map from CallDescriptions to arbitrary data. Provides a unified
/// way for checkers to react on function calls.
template <typename T> class CallDescriptionMap {
  friend class CallDescriptionSet;

  // Some call descriptions aren't easily hashable (eg., the ones with qualified
  // names in which some sections are omitted), so let's put them
  // in a simple vector and use linear lookup.
  // TODO: Implement an actual map for fast lookup for "hashable" call
  // descriptions (eg., the ones for C functions that just match the name).
  std::vector<std::pair<CallDescription, T>> LinearMap;

public:
  CallDescriptionMap(
      std::initializer_list<std::pair<CallDescription, T>> &&List)
      : LinearMap(List) {}

  template <typename InputIt>
  CallDescriptionMap(InputIt First, InputIt Last) : LinearMap(First, Last) {}

  ~CallDescriptionMap() = default;

  // These maps are usually stored once per checker, so let's make sure
  // we don't do redundant copies.
  CallDescriptionMap(const CallDescriptionMap &) = delete;
  CallDescriptionMap &operator=(const CallDescription &) = delete;

  CallDescriptionMap(CallDescriptionMap &&) = default;
  CallDescriptionMap &operator=(CallDescriptionMap &&) = default;

  LLVM_NODISCARD const T *lookup(const CallEvent &Call) const {
    // Slow path: linear lookup.
    // TODO: Implement some sort of fast path.
    for (const std::pair<CallDescription, T> &I : LinearMap)
      if (I.first.matches(Call))
        return &I.second;

    return nullptr;
  }
};

/// An immutable set of CallDescriptions.
/// Checkers can efficiently decide if a given CallEvent matches any
/// CallDescription in the set.
class CallDescriptionSet {
  CallDescriptionMap<bool /*unused*/> Impl = {};

public:
  CallDescriptionSet(std::initializer_list<CallDescription> &&List);

  CallDescriptionSet(const CallDescriptionSet &) = delete;
  CallDescriptionSet &operator=(const CallDescription &) = delete;

  LLVM_NODISCARD bool contains(const CallEvent &Call) const;
};

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CALLDESCRIPTION_H
