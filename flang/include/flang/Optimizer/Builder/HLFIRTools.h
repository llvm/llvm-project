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

namespace mlir {
class IRMapping;
}

namespace hlfir {

class AssociateOp;
class ElementalOp;
class ElementalOpInterface;
class ElementalAddrOp;
class EvaluateInMemoryOp;
class YieldElementOp;

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
  bool isProcedurePointer() const {
    return fir::isBoxProcAddressType(getType());
  }
  bool isBoxAddressOrValue() const {
    return hlfir::isBoxAddressOrValueType(getType());
  }

  /// Is this entity a procedure designator?
  bool isProcedure() const { return isFortranProcedureValue(getType()); }

  /// Is this an array or an assumed ranked entity?
  bool isArray() const { return getRank() != 0; }

  /// Is this an assumed ranked entity?
  bool isAssumedRank() const { return getRank() == -1; }

  /// Return the rank of this entity or -1 if it is an assumed rank.
  int getRank() const {
    mlir::Type type = fir::unwrapPassByRefType(fir::unwrapRefType(getType()));
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(type)) {
      if (seqTy.hasUnknownShape())
        return -1;
      return seqTy.getDimension();
    }
    if (auto exprType = mlir::dyn_cast<hlfir::ExprType>(type))
      return exprType.getRank();
    return 0;
  }
  bool isScalar() const { return !isArray(); }

  bool isPolymorphic() const { return hlfir::isPolymorphicType(getType()); }

  mlir::Type getFortranElementType() const {
    return hlfir::getFortranElementType(getType());
  }
  mlir::Type getElementOrSequenceType() const {
    return hlfir::getFortranElementOrSequenceType(getType());
  }

  bool hasLengthParameters() const {
    mlir::Type eleTy = getFortranElementType();
    return mlir::isa<fir::CharacterType>(eleTy) ||
           fir::isRecordWithTypeParameters(eleTy);
  }

  bool isCharacter() const {
    return mlir::isa<fir::CharacterType>(getFortranElementType());
  }

  bool hasIntrinsicType() const {
    mlir::Type eleTy = getFortranElementType();
    return fir::isa_trivial(eleTy) || mlir::isa<fir::CharacterType>(eleTy);
  }

  bool isDerivedWithLengthParameters() const {
    return fir::isRecordWithTypeParameters(getFortranElementType());
  }

  bool mayHaveNonDefaultLowerBounds() const;

  // Is this entity known to be contiguous at compile time?
  // Note that when this returns false, the entity may still
  // turn-out to be contiguous at runtime.
  bool isSimplyContiguous() const {
    // If this can be described without a fir.box in FIR, this must
    // be contiguous.
    if (!hlfir::isBoxAddressOrValueType(getFirBase().getType()))
      return true;
    // Otherwise, if this entity has a visible declaration in FIR,
    // or is the dereference of an allocatable or contiguous pointer
    // it is simply contiguous.
    if (auto varIface = getMaybeDereferencedVariableInterface())
      return varIface.isAllocatable() || varIface.hasContiguousAttr();
    return false;
  }

  fir::FortranVariableOpInterface getIfVariableInterface() const {
    return this->getDefiningOp<fir::FortranVariableOpInterface>();
  }

  // Return a "declaration" operation for this variable if visible,
  // or the "declaration" operation of the allocatable/pointer this
  // variable was dereferenced from (if it is visible).
  fir::FortranVariableOpInterface
  getMaybeDereferencedVariableInterface() const {
    mlir::Value base = *this;
    if (auto loadOp = base.getDefiningOp<fir::LoadOp>())
      base = loadOp.getMemref();
    return base.getDefiningOp<fir::FortranVariableOpInterface>();
  }

  bool isOptional() const {
    auto varIface = getIfVariableInterface();
    return varIface ? varIface.isOptional() : false;
  }

  bool isParameter() const {
    auto varIface = getIfVariableInterface();
    return varIface ? varIface.isParameter() : false;
  }

  bool isAllocatable() const {
    auto varIface = getIfVariableInterface();
    return varIface ? varIface.isAllocatable() : false;
  }

  bool isPointer() const {
    auto varIface = getIfVariableInterface();
    return varIface ? varIface.isPointer() : false;
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
                         Entity entity, bool contiguousHint = false);

/// Function to translate FortranVariableOpInterface to fir::ExtendedValue.
/// It may generates IR to unbox fir.boxchar, but has otherwise no side effects
/// on the IR.
fir::ExtendedValue
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         fir::FortranVariableOpInterface fortranVariable,
                         bool forceHlfirBase = false);

/// Generate declaration for a fir::ExtendedValue in memory.
fir::FortranVariableOpInterface
genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
           const fir::ExtendedValue &exv, llvm::StringRef name,
           fir::FortranVariableFlagsAttr flags,
           mlir::Value dummyScope = nullptr,
           cuf::DataAttributeAttr dataAttr = {});

/// Generate an hlfir.associate to build a variable from an expression value.
/// The type of the variable must be provided so that scalar logicals are
/// properly typed when placed in memory.
hlfir::AssociateOp
genAssociateExpr(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value, mlir::Type variableType,
                 llvm::StringRef name,
                 std::optional<mlir::NamedAttribute> attr = std::nullopt);

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

/// Get or create a fir.box or fir.class from a variable.
hlfir::Entity genVariableBox(mlir::Location loc, fir::FirOpBuilder &builder,
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

/// Generate lower bounds from a shape. If \p shape is null or is a fir.shape,
/// the returned vector will contain \p rank ones.
llvm::SmallVector<mlir::Value> genLowerbounds(mlir::Location loc,
                                              fir::FirOpBuilder &builder,
                                              mlir::Value shape, unsigned rank);

/// Compute fir.shape<> (no lower bounds) for an entity.
mlir::Value genShape(mlir::Location loc, fir::FirOpBuilder &builder,
                     Entity entity);

/// Compute the extent of \p entity in dimension \p dim. Crashes
/// if dim is bigger than the entity's rank.
mlir::Value genExtent(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity entity, unsigned dim);

/// Compute the lower bound of \p entity in dimension \p dim. Crashes
/// if dim is bigger than the entity's rank.
mlir::Value genLBound(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity entity, unsigned dim);

/// Generate a vector of extents with index type from a fir.shape
/// of fir.shape_shift value.
llvm::SmallVector<mlir::Value> getIndexExtents(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               mlir::Value shape);

/// Return explicit extents. If the base is a fir.box, this won't read it to
/// return the extents and will instead return an empty vector.
llvm::SmallVector<mlir::Value>
getExplicitExtentsFromShape(mlir::Value shape, fir::FirOpBuilder &builder);

/// Read length parameters into result if this entity has any.
void genLengthParameters(mlir::Location loc, fir::FirOpBuilder &builder,
                         Entity entity,
                         llvm::SmallVectorImpl<mlir::Value> &result);

/// Get the length of a character entity. Crashes if the entity is not
/// a character entity.
mlir::Value genCharLength(mlir::Location loc, fir::FirOpBuilder &builder,
                          Entity entity);

mlir::Value genRank(mlir::Location loc, fir::FirOpBuilder &builder,
                    Entity entity, mlir::Type resultType);

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
/// Get the entity type for an element of an array entity. Returns the
/// input type if it is a scalar. If the entity is a variable, this
/// is like getVariableElementType, otherwise, this will return a value
/// type (that may be an hlfir.expr type).
mlir::Type getEntityElementType(hlfir::Entity entity);

using ElementalKernelGenerator = std::function<hlfir::Entity(
    mlir::Location, fir::FirOpBuilder &, mlir::ValueRange)>;
/// Generate an hlfir.elementalOp given call back to generate the element
/// value at for each iteration.
/// If exprType is specified, this will be the return type of the elemental op.
/// If exprType is not specified, the resulting expression type is computed
/// from the given \p elementType and \p shape, and the type is polymorphic
/// if \p polymorphicMold is present.
hlfir::ElementalOp genElementalOp(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::Type elementType,
    mlir::Value shape, mlir::ValueRange typeParams,
    const ElementalKernelGenerator &genKernel, bool isUnordered = false,
    mlir::Value polymorphicMold = {}, mlir::Type exprType = mlir::Type{});

/// Structure to describe a loop nest.
struct LoopNest {
  mlir::Operation *outerOp = nullptr;
  mlir::Block *body = nullptr;
  llvm::SmallVector<mlir::Value> oneBasedIndices;
};

/// Generate a fir.do_loop nest looping from 1 to extents[i].
/// \p isUnordered specifies whether the loops in the loop nest
/// are unordered.
///
/// NOTE: genLoopNestWithReductions() should be used in favor
/// of this method, though, it cannot generate OpenMP workshare
/// loop constructs currently.
LoopNest genLoopNest(mlir::Location loc, fir::FirOpBuilder &builder,
                     mlir::ValueRange extents, bool isUnordered = false,
                     bool emitWorkshareLoop = false);
inline LoopNest genLoopNest(mlir::Location loc, fir::FirOpBuilder &builder,
                            mlir::Value shape, bool isUnordered = false,
                            bool emitWorkshareLoop = false) {
  return genLoopNest(loc, builder, getIndexExtents(loc, builder, shape),
                     isUnordered, emitWorkshareLoop);
}

/// The type of a callback that generates the body of a reduction
/// loop nest. It takes a location and a builder, as usual.
/// In addition, the first set of values are the values of the loops'
/// induction variables. The second set of values are the values
/// of the reductions on entry to the innermost loop.
/// The callback must return the updated values of the reductions.
using ReductionLoopBodyGenerator = std::function<llvm::SmallVector<mlir::Value>(
    mlir::Location, fir::FirOpBuilder &, mlir::ValueRange, mlir::ValueRange)>;

/// Generate a loop nest loopong from 1 to \p extents[i] and reducing
/// a set of values.
/// \p isUnordered specifies whether the loops in the loop nest
/// are unordered.
/// \p reductionInits are the initial values of the reductions
/// on entry to the outermost loop.
/// \p genBody callback is repsonsible for generating the code
/// that updates the reduction values in the innermost loop.
///
/// NOTE: the implementation of this function may decide
/// to perform the reductions on SSA or in memory.
/// In the latter case, this function is responsible for
/// allocating/loading/storing the reduction variables,
/// and making sure they have proper data sharing attributes
/// in case any parallel constructs are present around the point
/// of the loop nest insertion, or if the function decides
/// to use any worksharing loop constructs for the loop nest.
llvm::SmallVector<mlir::Value> genLoopNestWithReductions(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::ValueRange extents,
    mlir::ValueRange reductionInits, const ReductionLoopBodyGenerator &genBody,
    bool isUnordered = false);

/// Inline the body of an hlfir.elemental at the current insertion point
/// given a list of one based indices. This generates the computation
/// of one element of the elemental expression. Return the YieldElementOp
/// whose value argument is the element value.
/// The original hlfir::ElementalOp is left untouched.
hlfir::YieldElementOp inlineElementalOp(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        hlfir::ElementalOp elemental,
                                        mlir::ValueRange oneBasedIndices);

/// Inline the body of an hlfir.elemental or hlfir.elemental_addr without
/// cloning the resulting hlfir.yield_element/hlfir.yield, and return the cloned
/// operand of the hlfir.yield_element/hlfir.yield. The mapper must be provided
/// to cover complex cases where the inlined elemental is not defined in the
/// current context and uses values that have been cloned already. A callback is
/// provided to indicate if an hlfir.apply inside the hlfir.elemental must be
/// immediately replaced by the inlining of the applied hlfir.elemental.
mlir::Value inlineElementalOp(
    mlir::Location loc, fir::FirOpBuilder &builder,
    hlfir::ElementalOpInterface elemental, mlir::ValueRange oneBasedIndices,
    mlir::IRMapping &mapper,
    const std::function<bool(hlfir::ElementalOp)> &mustRecursivelyInline);

/// Create a new temporary with the shape and parameters of the provided
/// hlfir.eval_in_mem operation and clone the body of the hlfir.eval_in_mem
/// operating on this new temporary.  returns the temporary and whether the
/// temporary is heap or stack allocated.
std::pair<hlfir::Entity, bool>
computeEvaluateOpInNewTemp(mlir::Location, fir::FirOpBuilder &,
                           hlfir::EvaluateInMemoryOp evalInMem,
                           mlir::Value shape, mlir::ValueRange typeParams);

// Clone the body of the hlfir.eval_in_mem operating on this the provided
// storage.  The provided storage must be a contiguous "raw" memory reference
// (not a fir.box) big enough to hold the value computed by hlfir.eval_in_mem.
// No runtime check is inserted by this utility to enforce that. It is also
// usually invalid to provide some storage that is already addressed directly
// or indirectly inside the hlfir.eval_in_mem body.
void computeEvaluateOpIn(mlir::Location, fir::FirOpBuilder &,
                         hlfir::EvaluateInMemoryOp, mlir::Value storage);

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
convertToValue(mlir::Location loc, fir::FirOpBuilder &builder,
               hlfir::Entity entity);

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
convertToAddress(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity entity, mlir::Type targetType);

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
convertToBox(mlir::Location loc, fir::FirOpBuilder &builder,
             hlfir::Entity entity, mlir::Type targetType);

/// Clone an hlfir.elemental_addr into an hlfir.elemental value.
hlfir::ElementalOp cloneToElementalOp(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::ElementalAddrOp elementalAddrOp);

/// Return true, if \p elemental must produce a temporary array,
/// for example, for the purpose of finalization. Note that such
/// ElementalOp's must be optimized with caution. For example,
/// completely inlining such ElementalOp into another one
/// would be incorrect.
bool elementalOpMustProduceTemp(hlfir::ElementalOp elemental);

std::pair<hlfir::Entity, mlir::Value>
createTempFromMold(mlir::Location loc, fir::FirOpBuilder &builder,
                   hlfir::Entity mold);

// TODO: this does not support polymorphic molds
hlfir::Entity createStackTempFromMold(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity mold);

hlfir::EntityWithAttributes convertCharacterKind(mlir::Location loc,
                                                 fir::FirOpBuilder &builder,
                                                 hlfir::Entity scalarChar,
                                                 int toKind);

/// Materialize an implicit Fortran type conversion from \p source to \p toType.
/// This is a no-op if the Fortran category and KIND of \p source are
/// the same as the one in \p toType. This is also a no-op if \p toType is an
/// unlimited polymorphic. For characters, this implies that a conversion is
/// only inserted in case of KIND mismatch (and not in case of length mismatch),
/// and that the resulting entity length is the same as the one from \p source.
/// It is valid to call this helper if \p source is an array. If a conversion is
/// inserted for arrays, a clean-up will be returned. If no conversion is
/// needed, the source is returned.
/// Beware that the resulting entity mlir type may not be toType: it will be a
/// Fortran entity with the same Fortran category and KIND.
/// If preserveLowerBounds is set, the returned entity will have the same lower
/// bounds as \p source.
std::pair<hlfir::Entity, std::optional<hlfir::CleanupFunction>>
genTypeAndKindConvert(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity source, mlir::Type toType,
                      bool preserveLowerBounds);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_HLFIRTOOLS_H
