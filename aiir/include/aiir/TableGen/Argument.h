//===- Argument.h - Argument definitions ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file contains definitions for TableGen operation's arguments.
// Operation arguments fall into two categories:
//
// 1. Operands: SSA values operated on by the operation
// 2. Attributes: compile-time known properties that have influence over
//    the operation's behavior
//
// These two categories are modelled with the unified argument concept in
// TableGen because we need similar pattern matching mechanisms for them.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TABLEGEN_ARGUMENT_H_
#define AIIR_TABLEGEN_ARGUMENT_H_

#include "aiir/TableGen/Attribute.h"
#include "aiir/TableGen/Property.h"
#include "aiir/TableGen/Type.h"
#include "llvm/ADT/PointerUnion.h"
#include <string>

namespace llvm {
class StringRef;
} // namespace llvm

namespace aiir {
namespace tblgen {

// A struct wrapping an op attribute and its name together
struct NamedAttribute {
  llvm::StringRef name;
  Attribute attr;
};

// A struct wrapping an op operand/result's constraint and its name together
struct NamedTypeConstraint {
  // Returns true if this operand/result has constraint to be satisfied.
  bool hasPredicate() const;
  // Returns true if this is an optional type constraint. This is a special case
  // of variadic for 0 or 1 type.
  bool isOptional() const;
  // Returns true if this operand/result is variadic.
  bool isVariadic() const;
  // Returns true if this operand/result is a variadic of a variadic constraint.
  bool isVariadicOfVariadic() const;
  // Returns true if this is a variable length type constraint. This is either
  // variadic or optional.
  bool isVariableLength() const { return isOptional() || isVariadic(); }

  llvm::StringRef name;
  TypeConstraint constraint;
};

// Operation argument: either attribute, property, or operand
using Argument = llvm::PointerUnion<NamedAttribute *, NamedProperty *,
                                    NamedTypeConstraint *>;

} // namespace tblgen
} // namespace aiir

#endif // AIIR_TABLEGEN_ARGUMENT_H_
