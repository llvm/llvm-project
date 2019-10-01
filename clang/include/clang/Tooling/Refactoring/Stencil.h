//===--- Stencil.h - Stencil class ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// /file
/// This file defines the *Stencil* abstraction: a code-generating object,
/// parameterized by named references to (bound) AST nodes.  Given a match
/// result, a stencil can be evaluated to a string of source code.
///
/// A stencil is similar in spirit to a format string: it is composed of a
/// series of raw text strings, references to nodes (the parameters) and helper
/// code-generation operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_STENCIL_H_
#define LLVM_CLANG_TOOLING_REFACTOR_STENCIL_H_

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring/RangeSelector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace clang {
namespace tooling {

/// A stencil is represented as a sequence of "parts" that can each individually
/// generate a code string based on a match result.  The different kinds of
/// parts include (raw) text, references to bound nodes and assorted operations
/// on bound nodes.
///
/// Users can create custom Stencil operations by implementing this interface.
class StencilPartInterface {
public:
  virtual ~StencilPartInterface() = default;

  /// Evaluates this part to a string and appends it to \c Result.  \c Result is
  /// undefined in the case of an error.
  virtual llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &Match,
                           std::string *Result) const = 0;

  virtual bool isEqual(const StencilPartInterface &other) const = 0;

  const void *typeId() const { return TypeId; }

protected:
  StencilPartInterface(const void *DerivedId) : TypeId(DerivedId) {}

  // Since this is an abstract class, copying/assigning only make sense for
  // derived classes implementing `clone()`.
  StencilPartInterface(const StencilPartInterface &) = default;
  StencilPartInterface &operator=(const StencilPartInterface &) = default;

  /// Unique identifier of the concrete type of this instance.  Supports safe
  /// downcasting.
  const void *TypeId;
};

/// A copyable facade for a std::unique_ptr<StencilPartInterface>. Copies result
/// in a copy of the underlying pointee object.
class StencilPart {
public:
  explicit StencilPart(std::shared_ptr<StencilPartInterface> Impl)
      : Impl(std::move(Impl)) {}

  /// See `StencilPartInterface::eval()`.
  llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &Match,
                   std::string *Result) const {
    return Impl->eval(Match, Result);
  }

  bool operator==(const StencilPart &Other) const {
    if (Impl == Other.Impl)
      return true;
    if (Impl == nullptr || Other.Impl == nullptr)
      return false;
    return Impl->isEqual(*Other.Impl);
  }

private:
  std::shared_ptr<StencilPartInterface> Impl;
};

/// A sequence of code fragments, references to parameters and code-generation
/// operations that together can be evaluated to (a fragment of) source code,
/// given a match result.
class Stencil {
public:
  Stencil() = default;

  /// Composes a stencil from a series of parts.
  template <typename... Ts> static Stencil cat(Ts &&... Parts) {
    Stencil S;
    S.Parts = {wrap(std::forward<Ts>(Parts))...};
    return S;
  }

  /// Appends data from a \p OtherStencil to this stencil.
  void append(Stencil OtherStencil);

  // Evaluates the stencil given a match result. Requires that the nodes in the
  // result includes any ids referenced in the stencil. References to missing
  // nodes will result in an invalid_argument error.
  llvm::Expected<std::string>
  eval(const ast_matchers::MatchFinder::MatchResult &Match) const;

  // Allow Stencils to operate as std::function, for compatibility with
  // Transformer's TextGenerator.
  llvm::Expected<std::string>
  operator()(const ast_matchers::MatchFinder::MatchResult &Result) const {
    return eval(Result);
  }

private:
  friend bool operator==(const Stencil &A, const Stencil &B);
  static StencilPart wrap(llvm::StringRef Text);
  static StencilPart wrap(RangeSelector Selector);
  static StencilPart wrap(StencilPart Part) { return Part; }

  std::vector<StencilPart> Parts;
};

inline bool operator==(const Stencil &A, const Stencil &B) {
  return A.Parts == B.Parts;
}

inline bool operator!=(const Stencil &A, const Stencil &B) { return !(A == B); }

// Functions for conveniently building stencils.
namespace stencil {
/// Convenience wrapper for Stencil::cat that can be imported with a using decl.
template <typename... Ts> Stencil cat(Ts &&... Parts) {
  return Stencil::cat(std::forward<Ts>(Parts)...);
}

/// \returns exactly the text provided.
StencilPart text(llvm::StringRef Text);

/// \returns the source corresponding to the selected range.
StencilPart selection(RangeSelector Selector);

/// \returns the source corresponding to the identified node.
/// FIXME: Deprecated. Write `selection(node(Id))` instead.
inline StencilPart node(llvm::StringRef Id) {
  return selection(tooling::node(Id));
}

/// Variant of \c node() that identifies the node as a statement, for purposes
/// of deciding whether to include any trailing semicolon.  Only relevant for
/// Expr nodes, which, by default, are *not* considered as statements.
/// \returns the source corresponding to the identified node, considered as a
/// statement.
/// FIXME: Deprecated. Write `selection(statement(Id))` instead.
inline StencilPart sNode(llvm::StringRef Id) {
  return selection(tooling::statement(Id));
}

/// For debug use only; semantics are not guaranteed.
///
/// \returns the string resulting from calling the node's print() method.
StencilPart dPrint(llvm::StringRef Id);
} // namespace stencil
} // namespace tooling
} // namespace clang
#endif // LLVM_CLANG_TOOLING_REFACTOR_STENCIL_H_
