//===--- DependenceFlags.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DEPENDENCEFLAGS_H
#define LLVM_CLANG_AST_DEPENDENCEFLAGS_H

#include "clang/Basic/BitmaskEnum.h"
#include "llvm/ADT/BitmaskEnum.h"
#include <cstdint>

namespace clang {
struct ExprDependenceScope {
  enum ExprDependence : uint8_t {
    UnexpandedPack = 1,
    Instantiation = 2,
    Type = 4,
    Value = 8,

    // clang extension: this expr contains or references an error, and is
    // considered dependent on how that error is resolved.
    Error = 16,

    None = 0,
    All = 31,

    TypeValue = Type | Value,
    TypeInstantiation = Type | Instantiation,
    ValueInstantiation = Value | Instantiation,
    TypeValueInstantiation = Type | Value | Instantiation,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Error)
  };
};
using ExprDependence = ExprDependenceScope::ExprDependence;
static constexpr unsigned ExprDependenceBits = 5;

struct TypeDependenceScope {
  enum TypeDependence : uint8_t {
    /// Whether this type contains an unexpanded parameter pack
    /// (for C++11 variadic templates)
    UnexpandedPack = 1,
    /// Whether this type somehow involves a template parameter, even
    /// if the resolution of the type does not depend on a template parameter.
    Instantiation = 2,
    /// Whether this type is a dependent type (C++ [temp.dep.type]).
    Dependent = 4,
    /// Whether this type is a variably-modified type (C99 6.7.5).
    VariablyModified = 8,

    // FIXME: add Error bit.

    None = 0,
    All = 15,

    DependentInstantiation = Dependent | Instantiation,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/VariablyModified)
  };
};
using TypeDependence = TypeDependenceScope::TypeDependence;
static constexpr unsigned TypeDependenceBits = 4;

#define LLVM_COMMON_DEPENDENCE(NAME)                                           \
  struct NAME##Scope {                                                         \
    enum NAME : uint8_t {                                                      \
      UnexpandedPack = 1,                                                      \
      Instantiation = 2,                                                       \
      Dependent = 4,                                                           \
                                                                               \
      None = 0,                                                                \
      DependentInstantiation = Dependent | Instantiation,                      \
      All = 7,                                                                 \
                                                                               \
      LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Dependent)                    \
    };                                                                         \
  };                                                                           \
  using NAME = NAME##Scope::NAME;                                              \
  static constexpr unsigned NAME##Bits = 3;

LLVM_COMMON_DEPENDENCE(NestedNameSpecifierDependence)
LLVM_COMMON_DEPENDENCE(TemplateNameDependence)
LLVM_COMMON_DEPENDENCE(TemplateArgumentDependence)
#undef LLVM_COMMON_DEPENDENCE

/// Computes dependencies of a reference with the name having template arguments
/// with \p TA dependencies.
inline ExprDependence toExprDependence(TemplateArgumentDependence TA) {
  auto D = ExprDependence::None;
  if (TA & TemplateArgumentDependence::UnexpandedPack)
    D |= ExprDependence::UnexpandedPack;
  if (TA & TemplateArgumentDependence::Instantiation)
    D |= ExprDependence::Instantiation;
  if (TA & TemplateArgumentDependence::Dependent)
    D |= ExprDependence::Type | ExprDependence::Value;
  return D;
}
inline ExprDependence toExprDependence(TypeDependence TD) {
  // This hack works because TypeDependence and TemplateArgumentDependence
  // share the same bit representation, apart from variably-modified.
  return toExprDependence(static_cast<TemplateArgumentDependence>(
      TD & ~TypeDependence::VariablyModified));
}
inline ExprDependence toExprDependence(NestedNameSpecifierDependence NSD) {
  // This hack works because TypeDependence and TemplateArgumentDependence
  // share the same bit representation.
  return toExprDependence(static_cast<TemplateArgumentDependence>(NSD)) &
         ~ExprDependence::TypeValue;
}
inline ExprDependence turnTypeToValueDependence(ExprDependence D) {
  // Type-dependent expressions are always be value-dependent, so we simply drop
  // type dependency.
  return D & ~ExprDependence::Type;
}
inline ExprDependence turnValueToTypeDependence(ExprDependence D) {
  // Type-dependent expressions are always be value-dependent.
  if (D & ExprDependence::Value)
    D |= ExprDependence::Type;
  return D;
}

// Returned type-dependence will never have VariablyModified set.
inline TypeDependence toTypeDependence(ExprDependence D) {
  // Supported bits all have the same representation.
  return static_cast<TypeDependence>(D & (ExprDependence::UnexpandedPack |
                                          ExprDependence::Instantiation |
                                          ExprDependence::Type));
}
inline TypeDependence toTypeDependence(NestedNameSpecifierDependence D) {
  // Supported bits all have the same representation.
  return static_cast<TypeDependence>(D);
}
inline TypeDependence toTypeDependence(TemplateNameDependence D) {
  // Supported bits all have the same representation.
  return static_cast<TypeDependence>(D);
}
inline TypeDependence toTypeDependence(TemplateArgumentDependence D) {
  // Supported bits all have the same representation.
  return static_cast<TypeDependence>(D);
}

inline NestedNameSpecifierDependence
toNestedNameSpecifierDependendence(TypeDependence D) {
  // This works because both classes share the same bit representation.
  return static_cast<NestedNameSpecifierDependence>(
      D & ~TypeDependence::VariablyModified);
}

inline TemplateArgumentDependence
toTemplateArgumentDependence(TypeDependence D) {
  // This works because both classes share the same bit representation.
  return static_cast<TemplateArgumentDependence>(
      D & ~TypeDependence::VariablyModified);
}
inline TemplateArgumentDependence
toTemplateArgumentDependence(TemplateNameDependence D) {
  // This works because both classes share the same bit representation.
  return static_cast<TemplateArgumentDependence>(D);
}
inline TemplateArgumentDependence
toTemplateArgumentDependence(ExprDependence ED) {
  TemplateArgumentDependence TAD = TemplateArgumentDependence::None;
  if (ED & (ExprDependence::Type | ExprDependence::Value))
    TAD |= TemplateArgumentDependence::Dependent;
  if (ED & ExprDependence::Instantiation)
    TAD |= TemplateArgumentDependence::Instantiation;
  if (ED & ExprDependence::UnexpandedPack)
    TAD |= TemplateArgumentDependence::UnexpandedPack;
  return TAD;
}

inline TemplateNameDependence
toTemplateNameDependence(NestedNameSpecifierDependence D) {
  return static_cast<TemplateNameDependence>(D);
}

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // namespace clang
#endif
