//===-- HLFIRTools.h -- HLFIR tools       -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
#define FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"

namespace fir {
class FirOpBuilder;
}

namespace hlfir {

/// Is this an SSA value type for the value of a Fortran expression?
inline bool isFortranValueType(mlir::Type type) {
  return type.isa<hlfir::ExprType>() || fir::isa_trivial(type);
}

/// Is this the value of a Fortran expression in an SSA value form?
inline bool isFortranValue(mlir::Value value) {
  return isFortranValueType(value.getType());
}

/// Is this a Fortran variable?
/// Note that by "variable", it must be understood that the mlir::Value is
/// a memory value of a storage that can be reason about as a Fortran object
/// (its bounds, shape, and type parameters, if any, are retrievable).
/// This does not imply that the mlir::Value points to a variable from the
/// original source or can be legally defined: temporaries created to store
/// expression values are considered to be variables, and so are PARAMETERs
/// global constant address.
inline bool isFortranEntity(mlir::Value value) {
  return isFortranValue(value) || isFortranVariableType(value.getType());
}

/// Is this a Fortran variable for which the defining op carrying the Fortran
/// attributes is visible?
inline bool isFortranVariableWithAttributes(mlir::Value value) {
  return value.getDefiningOp<fir::FortranVariableOpInterface>();
}

/// Is this a Fortran expression value, or a Fortran variable for which the
/// defining op carrying the Fortran attributes is visible?
inline bool isFortranEntityWithAttributes(mlir::Value value) {
  return isFortranValue(value) || isFortranVariableWithAttributes(value);
}

class Entity : public mlir::Value {
public:
  explicit Entity(mlir::Value value) : mlir::Value(value) {
    assert(isFortranEntity(value) &&
           "must be a value representing a Fortran value or variable like");
  }
  Entity(fir::FortranVariableOpInterface variable)
      : mlir::Value(variable.getBase()) {}
  bool isValue() const { return isFortranValue(*this); }
  bool isVariable() const { return !isValue(); }
  bool isMutableBox() const { return hlfir::isBoxAddressType(getType()); }
  bool isArray() const {
    mlir::Type type = fir::unwrapPassByRefType(fir::unwrapRefType(getType()));
    if (type.isa<fir::SequenceType>())
      return true;
    if (auto exprType = type.dyn_cast<hlfir::ExprType>())
      return exprType.isArray();
    return false;
  }
  bool isScalar() const { return !isArray(); }

  mlir::Type getFortranElementType() const {
    return hlfir::getFortranElementType(getType());
  }

  bool hasLengthParameters() const {
    mlir::Type eleTy = getFortranElementType();
    return eleTy.isa<fir::CharacterType>() ||
           fir::isRecordWithTypeParameters(eleTy);
  }

  fir::FortranVariableOpInterface getIfVariableInterface() const {
    return this->getDefiningOp<fir::FortranVariableOpInterface>();
  }

  // Get the entity as an mlir SSA value containing all the shape, type
  // parameters and dynamic shape information.
  mlir::Value getBase() const { return *this; }

  // Get the entity as a FIR base. This may not carry the shape and type
  // parameters information, and even when it is a box with shape information.
  // it will not contain the local lower bounds of the entity. This should
  // be used with care when generating FIR code that does not need this
  // information, or has access to it in other ways. Its advantage is that
  // it will never be a fir.box for explicit shape arrays, leading to simpler
  // FIR code generation.
  mlir::Value getFirBase() const;
};

/// Wrapper over an mlir::Value that can be viewed as a Fortran entity.
/// This provides some Fortran specific helpers as well as a guarantee
/// in the compiler source that a certain mlir::Value must be a Fortran
/// entity, and if it is a variable, its defining operation carrying its
/// Fortran attributes must be visible.
class EntityWithAttributes : public Entity {
public:
  explicit EntityWithAttributes(mlir::Value value) : Entity(value) {
    assert(isFortranEntityWithAttributes(value) &&
           "must be a value representing a Fortran value or variable");
  }
  EntityWithAttributes(fir::FortranVariableOpInterface variable)
      : Entity(variable) {}
  fir::FortranVariableOpInterface getIfVariable() const {
    return getIfVariableInterface();
  }
};

/// Functions to translate hlfir::EntityWithAttributes to fir::ExtendedValue.
/// For Fortran arrays, character, and derived type values, this require
/// allocating a storage since these can only be represented in memory in FIR.
/// In that case, a cleanup function is provided to generate the finalization
/// code after the end of the fir::ExtendedValue use.
using CleanupFunction = std::function<void()>;
std::pair<fir::ExtendedValue, llvm::Optional<CleanupFunction>>
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity);

/// Function to translate FortranVariableOpInterface to fir::ExtendedValue.
/// It does not generate any IR, and is a simple packaging operation.
fir::ExtendedValue
translateToExtendedValue(fir::FortranVariableOpInterface fortranVariable);

/// Generate declaration for a fir::ExtendedValue in memory.
EntityWithAttributes genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
                                const fir::ExtendedValue &exv,
                                llvm::StringRef name,
                                fir::FortranVariableFlagsAttr flags);

/// If the entity is a variable, load its value (dereference pointers and
/// allocatables if needed). Do nothing if the entity os already a variable or
/// if it is not a scalar entity of numerical or logical type.
Entity loadTrivialScalar(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
