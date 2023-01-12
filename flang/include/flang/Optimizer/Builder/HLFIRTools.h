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
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include <optional>

namespace fir {
class FirOpBuilder;
}

namespace hlfir {

class AssociateOp;
class ElementalOp;
class YieldElementOp;

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
  bool isBoxAddressOrValue() const {
    return hlfir::isBoxAddressOrValueType(getType());
  }
  bool isArray() const {
    mlir::Type type = fir::unwrapPassByRefType(fir::unwrapRefType(getType()));
    if (type.isa<fir::SequenceType>())
      return true;
    if (auto exprType = type.dyn_cast<hlfir::ExprType>())
      return exprType.isArray();
    return false;
  }
  bool isScalar() const { return !isArray(); }

  bool isPolymorphic() const {
    if (auto exprType = getType().dyn_cast<hlfir::ExprType>())
      return exprType.isPolymorphic();
    return fir::isPolymorphicType(getType());
  }

  mlir::Type getFortranElementType() const {
    return hlfir::getFortranElementType(getType());
  }

  bool hasLengthParameters() const {
    mlir::Type eleTy = getFortranElementType();
    return eleTy.isa<fir::CharacterType>() ||
           fir::isRecordWithTypeParameters(eleTy);
  }

  bool isCharacter() const {
    return getFortranElementType().isa<fir::CharacterType>();
  }

  bool hasNonDefaultLowerBounds() const {
    if (!isBoxAddressOrValue() || isScalar())
      return false;
    if (isMutableBox())
      return true;
    if (auto varIface = getIfVariableInterface())
      if (auto shape = varIface.getShape()) {
        auto shapeTy = shape.getType();
        return shapeTy.isa<fir::ShiftType>() ||
               shapeTy.isa<fir::ShapeShiftType>();
      }
    return true;
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
std::pair<fir::ExtendedValue, std::optional<CleanupFunction>>
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity);

/// Function to translate FortranVariableOpInterface to fir::ExtendedValue.
/// It may generates IR to unbox fir.boxchar, but has otherwise no side effects
/// on the IR.
fir::ExtendedValue
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         fir::FortranVariableOpInterface fortranVariable);

/// Generate declaration for a fir::ExtendedValue in memory.
fir::FortranVariableOpInterface genDeclare(mlir::Location loc,
                                           fir::FirOpBuilder &builder,
                                           const fir::ExtendedValue &exv,
                                           llvm::StringRef name,
                                           fir::FortranVariableFlagsAttr flags);

/// Generate an hlfir.associate to build a variable from an expression value.
/// The type of the variable must be provided so that scalar logicals are
/// properly typed when placed in memory.
hlfir::AssociateOp genAssociateExpr(mlir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity value,
                                    mlir::Type variableType,
                                    llvm::StringRef name);

/// Get the raw address of a variable (simple fir.ref/fir.ptr, or fir.heap
/// value). The returned value should be used with care, it does not contain any
/// stride, shape, and type parameter information. For pointers and
/// allocatables, this returns the address of the target.
mlir::Value genVariableRawAddress(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  hlfir::Entity var);

/// Get a fir.boxchar for character scalar or array variable (the shape is lost
/// for arrays).
mlir::Value genVariableBoxChar(mlir::Location loc, fir::FirOpBuilder &builder,
                               hlfir::Entity var);

/// If the entity is a variable, load its value (dereference pointers and
/// allocatables if needed). Do nothing if the entity is already a value, and
/// only dereference pointers and allocatables if it is not a scalar entity
/// of numerical or logical type.
Entity loadTrivialScalar(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity);

/// If \p entity is a POINTER or ALLOCATABLE, dereference it and return the
/// target entity. Return \p entity otherwise.
hlfir::Entity derefPointersAndAllocatables(mlir::Location loc,
                                           fir::FirOpBuilder &builder,
                                           Entity entity);

/// Get element entity(oneBasedIndices) if entity is an array, or return entity
/// if it is a scalar. The indices are one based. If the entity has non default
/// lower bounds, the function will adapt the indices in the indexing operation.
hlfir::Entity getElementAt(mlir::Location loc, fir::FirOpBuilder &builder,
                           Entity entity, mlir::ValueRange oneBasedIndices);
/// Compute the lower and upper bounds of an entity.
llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
genBounds(mlir::Location loc, fir::FirOpBuilder &builder, Entity entity);
/// Compute the lower and upper bounds given a fir.shape or fir.shape_shift
/// (fir.shift is not allowed here).
llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
genBounds(mlir::Location loc, fir::FirOpBuilder &builder, mlir::Value shape);

/// Compute fir.shape<> (no lower bounds) for an entity.
mlir::Value genShape(mlir::Location loc, fir::FirOpBuilder &builder,
                     Entity entity);

/// Generate a vector of extents with index type from a fir.shape
/// of fir.shape_shift value.
llvm::SmallVector<mlir::Value> getIndexExtents(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               mlir::Value shape);

/// Read length parameters into result if this entity has any.
void genLengthParameters(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity,
                         llvm::SmallVectorImpl<mlir::Value> &result);

/// Get the length of a character entity. Crashes if the entity is not
/// a character entity.
mlir::Value genCharLength(mlir::Location loc, fir::FirOpBuilder &builder,
                          Entity entity);

/// Return the fir base, shape, and type parameters for a variable. Note that
/// type parameters are only added if the entity is not a box and the type
/// parameters is not a constant in the base type. This matches the arguments
/// expected by fir.embox/fir.array_coor.
std::pair<mlir::Value, mlir::Value> genVariableFirBaseShapeAndParams(
    mlir::Location loc, fir::FirOpBuilder &builder, Entity entity,
    llvm::SmallVectorImpl<mlir::Value> &typeParams);

/// Get the variable type for an element of an array type entity. Returns the
/// input entity type if it is scalar. Will crash if the entity is not a
/// variable.
mlir::Type getVariableElementType(hlfir::Entity variable);

using ElementalKernelGenerator = std::function<hlfir::Entity(
    mlir::Location, fir::FirOpBuilder &, mlir::ValueRange)>;
/// Generate an hlfir.elementalOp given call back to generate the element
/// value at for each iteration.
hlfir::ElementalOp genElementalOp(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  mlir::Type elementType, mlir::Value shape,
                                  mlir::ValueRange typeParams,
                                  const ElementalKernelGenerator &genKernel);

/// Generate a fir.do_loop nest looping from 1 to extents[i].
/// Return the inner fir.do_loop and the indices of the loops.
std::pair<fir::DoLoopOp, llvm::SmallVector<mlir::Value>>
genLoopNest(mlir::Location loc, fir::FirOpBuilder &builder,
            mlir::ValueRange extents);
inline std::pair<fir::DoLoopOp, llvm::SmallVector<mlir::Value>>
genLoopNest(mlir::Location loc, fir::FirOpBuilder &builder, mlir::Value shape) {
  return genLoopNest(loc, builder, getIndexExtents(loc, builder, shape));
}

/// Inline the body of an hlfir.elemental at the current insertion point
/// given a list of one based indices. This generates the computation
/// of one element of the elemental expression. Return the YieldElementOp
/// whose value argument is the element value.
/// The original hlfir::ElementalOp is left untouched.
hlfir::YieldElementOp inlineElementalOp(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        hlfir::ElementalOp elemental,
                                        mlir::ValueRange oneBasedIndices);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
