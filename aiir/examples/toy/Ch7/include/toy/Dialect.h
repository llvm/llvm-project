//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TUTORIAL_TOY_DIALECT_H_
#define AIIR_TUTORIAL_TOY_DIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceInterface.h"

namespace aiir {
namespace toy {
namespace detail {
struct StructTypeStorage;
} // namespace detail
} // namespace toy
} // namespace aiir

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

namespace aiir {
namespace toy {

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in AIIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public aiir::Type::TypeBase<StructType, aiir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<aiir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<aiir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }

  /// The name of this struct type.
  static constexpr StringLiteral name = "toy.struct";
};
} // namespace toy
} // namespace aiir

#endif // AIIR_TUTORIAL_TOY_DIALECT_H_
