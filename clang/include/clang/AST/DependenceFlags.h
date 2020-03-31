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

    /// Whether this type references an error, e.g. decltype(err-expression)
    /// yields an error type.
    Error = 16,

    None = 0,
    All = 31,

    DependentInstantiation = Dependent | Instantiation,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Error)
  };
};
using TypeDependence = TypeDependenceScope::TypeDependence;

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
  using NAME = NAME##Scope::NAME;

LLVM_COMMON_DEPENDENCE(NestedNameSpecifierDependence)
LLVM_COMMON_DEPENDENCE(TemplateNameDependence)
LLVM_COMMON_DEPENDENCE(TemplateArgumentDependence)
#undef LLVM_COMMON_DEPENDENCE

// A combined space of all dependence concepts for all node types.
// Used when aggregating dependence of nodes of different types.
class Dependence {
public:
  enum Bits : uint8_t {
    None = 0,

    // Contains a template parameter pack that wasn't expanded.
    UnexpandedPack = 1,
    // Uses a template parameter, even if it doesn't affect the result.
    // Validity depends on the template parameter.
    Instantiation = 2,
    // Expression type depends on template context.
    // Value and Instantiation should also be set.
    Type = 4,
    // Expression value depends on template context.
    // Instantiation should also be set.
    Value = 8,
    // Depends on template context.
    // The type/value distinction is only meaningful for expressions.
    Dependent = Type | Value,
    // Includes an error, and depends on how it is resolved.
    Error = 16,
    // Type depends on a runtime value (variable-length array).
    VariablyModified = 32,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/VariablyModified)
  };

  Dependence() : V(None) {}

  Dependence(TypeDependence D)
      : V(translate(D, TypeDependence::UnexpandedPack, UnexpandedPack) |
             translate(D, TypeDependence::Instantiation, Instantiation) |
             translate(D, TypeDependence::Dependent, Dependent) |
             translate(D, TypeDependence::VariablyModified, VariablyModified)) {
  }

  Dependence(ExprDependence D)
      : V(translate(D, ExprDependence::UnexpandedPack, UnexpandedPack) |
             translate(D, ExprDependence::Instantiation, Instantiation) |
             translate(D, ExprDependence::Type, Type) |
             translate(D, ExprDependence::Value, Value) |
             translate(D, ExprDependence::Error, Error)) {}

  Dependence(NestedNameSpecifierDependence D) :
    V ( translate(D, NNSDependence::UnexpandedPack, UnexpandedPack) |
            translate(D, NNSDependence::Instantiation, Instantiation) |
            translate(D, NNSDependence::Dependent, Dependent)){}

  Dependence(TemplateArgumentDependence D)
      : V(translate(D, TADependence::UnexpandedPack, UnexpandedPack) |
             translate(D, TADependence::Instantiation, Instantiation) |
             translate(D, TADependence::Dependent, Dependent)) {}

  Dependence(TemplateNameDependence D)
      : V(translate(D, TNDependence::UnexpandedPack, UnexpandedPack) |
             translate(D, TNDependence::Instantiation, Instantiation) |
             translate(D, TNDependence::Dependent, Dependent)) {}

  TypeDependence type() const {
    return translate(V, UnexpandedPack, TypeDependence::UnexpandedPack) |
           translate(V, Instantiation, TypeDependence::Instantiation) |
           translate(V, Dependent, TypeDependence::Dependent) |
           translate(V, Error, TypeDependence::Error) |
           translate(V, VariablyModified, TypeDependence::VariablyModified);
  }

  ExprDependence expr() const {
    return translate(V, UnexpandedPack, ExprDependence::UnexpandedPack) |
           translate(V, Instantiation, ExprDependence::Instantiation) |
           translate(V, Type, ExprDependence::Type) |
           translate(V, Value, ExprDependence::Value) |
           translate(V, Error, ExprDependence::Error);
  }

  NestedNameSpecifierDependence nestedNameSpecifier() const {
    return translate(V, UnexpandedPack, NNSDependence::UnexpandedPack) |
           translate(V, Instantiation, NNSDependence::Instantiation) |
           translate(V, Dependent, NNSDependence::Dependent);
  }

  TemplateArgumentDependence templateArgument() const {
    return translate(V, UnexpandedPack, TADependence::UnexpandedPack) |
           translate(V, Instantiation, TADependence::Instantiation) |
           translate(V, Dependent, TADependence::Dependent);
  }

  TemplateNameDependence templateName() const {
    return translate(V, UnexpandedPack, TNDependence::UnexpandedPack) |
           translate(V, Instantiation, TNDependence::Instantiation) |
           translate(V, Dependent, TNDependence::Dependent);
  }

private:
  Bits V;

  template <typename T, typename U>
  static U translate(T Bits, T FromBit, U ToBit) {
    return (Bits & FromBit) ? ToBit : static_cast<U>(0);
  }

  // Abbreviations to make conversions more readable.
  using NNSDependence = NestedNameSpecifierDependence;
  using TADependence = TemplateArgumentDependence;
  using TNDependence = TemplateNameDependence;
};

/// Computes dependencies of a reference with the name having template arguments
/// with \p TA dependencies.
inline ExprDependence toExprDependence(TemplateArgumentDependence TA) {
  return Dependence(TA).expr();
}
inline ExprDependence toExprDependence(TypeDependence D) {
  return Dependence(D).expr();
}
// Note: it's often necessary to strip `Dependent` from qualifiers.
// If V<T>:: refers to the current instantiation, NNS is considered dependent
// but the containing V<T>::foo likely isn't.
inline ExprDependence toExprDependence(NestedNameSpecifierDependence D) {
  return Dependence(D).expr();
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
  return Dependence(D).type();
}
inline TypeDependence toTypeDependence(NestedNameSpecifierDependence D) {
  return Dependence(D).type();
}
inline TypeDependence toTypeDependence(TemplateNameDependence D) {
  return Dependence(D).type();
}
inline TypeDependence toTypeDependence(TemplateArgumentDependence D) {
  return Dependence(D).type();
}

inline NestedNameSpecifierDependence
toNestedNameSpecifierDependendence(TypeDependence D) {
  return Dependence(D).nestedNameSpecifier();
}

inline TemplateArgumentDependence
toTemplateArgumentDependence(TypeDependence D) {
  return Dependence(D).templateArgument();
}
inline TemplateArgumentDependence
toTemplateArgumentDependence(TemplateNameDependence D) {
  return Dependence(D).templateArgument();
}
inline TemplateArgumentDependence
toTemplateArgumentDependence(ExprDependence D) {
  return Dependence(D).templateArgument();
}

inline TemplateNameDependence
toTemplateNameDependence(NestedNameSpecifierDependence D) {
  return Dependence(D).templateName();
}

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // namespace clang
#endif
