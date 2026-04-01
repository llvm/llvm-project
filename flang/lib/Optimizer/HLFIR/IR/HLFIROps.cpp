//===-- HLFIROps.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include <iterator>
#include <aiir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <tuple>

static llvm::cl::opt<bool> useStrictIntrinsicVerifier(
    "strict-intrinsic-verifier", llvm::cl::init(false),
    llvm::cl::desc("use stricter verifier for HLFIR intrinsic operations"));

/// generic implementation of the memory side effects interface for hlfir
/// transformational intrinsic operations
static void
getIntrinsicEffects(aiir::Operation *self,
                    llvm::SmallVectorImpl<aiir::SideEffects::EffectInstance<
                        aiir::MemoryEffects::Effect>> &effects) {
  // allocation effect if we return an expr
  assert(self->getNumResults() == 1 &&
         "hlfir intrinsic ops only produce 1 result");
  if (aiir::isa<hlfir::ExprType>(self->getResult(0).getType()))
    effects.emplace_back(aiir::MemoryEffects::Allocate::get(),
                         self->getOpResult(0),
                         aiir::SideEffects::DefaultResource::get());

  // read effect if we read from a pointer or refference type
  // or a box who'se pointer is read from inside of the intrinsic so that
  // loop conflicts can be detected in code like
  // hlfir.region_assign {
  //   %2 = hlfir.transpose %0#0 : (!fir.box<!fir.array<?x?xf32>>) ->
  //   !hlfir.expr<?x?xf32> hlfir.yield %2 : !hlfir.expr<?x?xf32> cleanup {
  //     hlfir.destroy %2 : !hlfir.expr<?x?xf32>
  //   }
  // } to {
  //   hlfir.yield %0#0 : !fir.box<!fir.array<?x?xf32>>
  // }
  for (aiir::OpOperand &operand : self->getOpOperands()) {
    aiir::Type opTy = operand.get().getType();
    fir::addVolatileMemoryEffects({opTy}, effects);
    if (fir::isa_ref_type(opTy) || fir::isa_box_type(opTy))
      effects.emplace_back(aiir::MemoryEffects::Read::get(), &operand,
                           aiir::SideEffects::DefaultResource::get());
  }
}

/// Verification helper for checking if two types are the same.
/// Set \p allowCharacterLenMismatch to true, if character types
/// of different known lengths should be treated as the same.
template <typename Op>
static llvm::LogicalResult areMatchingTypes(Op &op, aiir::Type type1,
                                            aiir::Type type2,
                                            bool allowCharacterLenMismatch) {
  if (auto charType1 = aiir::dyn_cast<fir::CharacterType>(type1))
    if (auto charType2 = aiir::dyn_cast<fir::CharacterType>(type2)) {
      // Character kinds must match.
      if (charType1.getFKind() != charType2.getFKind())
        return op.emitOpError("character KIND mismatch");

      // Constant propagation can result in mismatching lengths
      // in the dead code, but we should not fail on this.
      if (!allowCharacterLenMismatch)
        if (charType1.getLen() != fir::CharacterType::unknownLen() &&
            charType2.getLen() != fir::CharacterType::unknownLen() &&
            charType1.getLen() != charType2.getLen())
          return op.emitOpError("character LEN mismatch");

      return aiir::success();
    }

  return type1 == type2 ? aiir::success() : aiir::failure();
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// Is this a fir.[ref/ptr/heap]<fir.[box/class]<fir.heap<T>>> type?
static bool isAllocatableBoxRef(aiir::Type type) {
  fir::BaseBoxType boxType =
      aiir::dyn_cast_or_null<fir::BaseBoxType>(fir::dyn_cast_ptrEleTy(type));
  return boxType && aiir::isa<fir::HeapType>(boxType.getEleTy());
}

llvm::LogicalResult hlfir::AssignOp::verify() {
  aiir::Type lhsType = getLhs().getType();
  if (isAllocatableAssignment() && !isAllocatableBoxRef(lhsType))
    return emitOpError("lhs must be an allocatable when `realloc` is set");
  if (mustKeepLhsLengthInAllocatableAssignment() &&
      !(isAllocatableAssignment() &&
        aiir::isa<fir::CharacterType>(hlfir::getFortranElementType(lhsType))))
    return emitOpError("`realloc` must be set and lhs must be a character "
                       "allocatable when `keep_lhs_length_if_realloc` is set");
  return aiir::success();
}

void hlfir::AssignOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  aiir::OpOperand &rhs = getRhsMutable();
  aiir::OpOperand &lhs = getLhsMutable();
  aiir::Type rhsType = getRhs().getType();
  aiir::Type lhsType = getLhs().getType();
  if (aiir::isa<fir::RecordType>(hlfir::getFortranElementType(lhsType))) {
    // For derived type assignments, set unknown read/write effects since it
    // is not known here if user defined finalization is needed, and also
    // because allocatable components may lead to "deeper" read/write effects
    // that cannot be described with this API.
    effects.emplace_back(aiir::MemoryEffects::Read::get(),
                         aiir::SideEffects::DefaultResource::get());
    effects.emplace_back(aiir::MemoryEffects::Write::get(),
                         aiir::SideEffects::DefaultResource::get());
  } else {
    // Read effect when RHS is a variable.
    if (hlfir::isFortranVariableType(rhsType)) {
      if (hlfir::isBoxAddressType(rhsType)) {
        // Unknown read effect if the RHS is a descriptor since the read effect
        // on the data cannot be described.
        effects.emplace_back(aiir::MemoryEffects::Read::get(),
                             aiir::SideEffects::DefaultResource::get());
      } else {
        effects.emplace_back(aiir::MemoryEffects::Read::get(), &rhs,
                             aiir::SideEffects::DefaultResource::get());
      }
    }

    // Write effects on LHS.
    if (hlfir::isBoxAddressType(lhsType)) {
      //  If the LHS is a descriptor, the descriptor will be read and the data
      //  write cannot be described in this API (and the descriptor may be
      //  written to in case of realloc, which is covered by the unknown write
      //  effect.
      effects.emplace_back(aiir::MemoryEffects::Read::get(), &lhs,
                           aiir::SideEffects::DefaultResource::get());
      effects.emplace_back(aiir::MemoryEffects::Write::get(),
                           aiir::SideEffects::DefaultResource::get());
    } else {
      effects.emplace_back(aiir::MemoryEffects::Write::get(), &lhs,
                           aiir::SideEffects::DefaultResource::get());
    }
  }

  fir::addVolatileMemoryEffects({lhsType, rhsType}, effects);

  if (getRealloc()) {
    // Reallocation of the data cannot be precisely described by this API.
    effects.emplace_back(aiir::MemoryEffects::Free::get(),
                         aiir::SideEffects::DefaultResource::get());
    effects.emplace_back(aiir::MemoryEffects::Allocate::get(),
                         aiir::SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

static std::pair<aiir::Type, aiir::Type>
getDeclareOutputTypes(aiir::Type inputType, bool hasExplicitLowerBounds) {
  // Drop pointer/allocatable attribute of descriptor values. Only descriptor
  // addresses are ALLOCATABLE/POINTER. The HLFIR box result of an hlfir.declare
  // without those attributes should not have these attributes set.
  if (auto baseBoxType = aiir::dyn_cast<fir::BaseBoxType>(inputType))
    if (baseBoxType.isPointerOrAllocatable()) {
      aiir::Type boxWithoutAttributes =
          baseBoxType.getBoxTypeWithNewAttr(fir::BaseBoxType::Attribute::None);
      return {boxWithoutAttributes, boxWithoutAttributes};
    }
  aiir::Type type = fir::unwrapRefType(inputType);
  if (aiir::isa<fir::BaseBoxType>(type))
    return {inputType, inputType};
  if (auto charType = aiir::dyn_cast<fir::CharacterType>(type))
    if (charType.hasDynamicLen()) {
      aiir::Type hlfirType =
          fir::BoxCharType::get(charType.getContext(), charType.getFKind());
      return {hlfirType, inputType};
    }

  auto seqType = aiir::dyn_cast<fir::SequenceType>(type);
  bool hasDynamicExtents =
      seqType && fir::sequenceWithNonConstantShape(seqType);
  aiir::Type eleType = seqType ? seqType.getEleTy() : type;
  bool hasDynamicLengthParams = fir::characterWithDynamicLen(eleType) ||
                                fir::isRecordWithTypeParameters(eleType);
  if (hasExplicitLowerBounds || hasDynamicExtents || hasDynamicLengthParams) {
    aiir::Type boxType =
        fir::BoxType::get(type, fir::isa_volatile_type(inputType));
    return {boxType, inputType};
  }
  return {inputType, inputType};
}

/// Given a FIR memory type, and information about non default lower bounds, get
/// the related HLFIR variable type.
aiir::Type hlfir::DeclareOp::getHLFIRVariableType(aiir::Type inputType,
                                                  bool hasExplicitLowerBounds) {
  return getDeclareOutputTypes(inputType, hasExplicitLowerBounds).first;
}

static bool hasExplicitLowerBounds(aiir::Value shape) {
  return shape &&
         aiir::isa<fir::ShapeShiftType, fir::ShiftType>(shape.getType());
}

static std::pair<aiir::Type, aiir::Value>
updateDeclaredInputTypeWithVolatility(aiir::Type inputType, aiir::Value memref,
                                      aiir::OpBuilder &builder,
                                      fir::FortranVariableFlagsEnum flags) {
  if (!bitEnumContainsAny(flags,
                          fir::FortranVariableFlagsEnum::fortran_volatile)) {
    return std::make_pair(inputType, memref);
  }

  // A volatile pointer's pointee is volatile.
  const bool isPointer =
      bitEnumContainsAny(flags, fir::FortranVariableFlagsEnum::pointer);
  // An allocatable's inner type's volatility matches that of the reference.
  const bool isAllocatable =
      bitEnumContainsAny(flags, fir::FortranVariableFlagsEnum::allocatable);

  auto updateType = [&](auto t) {
    using FIRT = decltype(t);
    auto elementType = t.getEleTy();
    const bool elementTypeIsBox = aiir::isa<fir::BaseBoxType>(elementType);
    const bool elementTypeIsVolatile = isPointer || isAllocatable ||
                                       elementTypeIsBox ||
                                       fir::isa_volatile_type(elementType);
    auto newEleTy =
        fir::updateTypeWithVolatility(elementType, elementTypeIsVolatile);
    inputType = FIRT::get(newEleTy, true);
  };
  llvm::TypeSwitch<aiir::Type>(inputType)
      .Case<fir::ReferenceType, fir::BoxType, fir::ClassType>(updateType);
  memref =
      fir::VolatileCastOp::create(builder, memref.getLoc(), inputType, memref);
  return std::make_pair(inputType, memref);
}

void hlfir::DeclareOp::build(
    aiir::OpBuilder &builder, aiir::OperationState &result, aiir::Value memref,
    llvm::StringRef uniq_name, aiir::Value shape, aiir::ValueRange typeparams,
    aiir::Value dummy_scope, aiir::Value storage, std::uint64_t storage_offset,
    fir::FortranVariableFlagsAttr fortran_attrs,
    cuf::DataAttributeAttr data_attr, unsigned dummy_arg_no) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  aiir::Type inputType = memref.getType();
  bool hasExplicitLbs = hasExplicitLowerBounds(shape);
  if (fortran_attrs) {
    const auto flags = fortran_attrs.getFlags();
    std::tie(inputType, memref) = updateDeclaredInputTypeWithVolatility(
        inputType, memref, builder, flags);
  }
  auto [hlfirVariableType, firVarType] =
      getDeclareOutputTypes(inputType, hasExplicitLbs);
  aiir::IntegerAttr argNoAttr;
  if (dummy_arg_no > 0)
    argNoAttr = builder.getUI32IntegerAttr(dummy_arg_no);
  build(builder, result, {hlfirVariableType, firVarType}, memref, shape,
        typeparams, dummy_scope, storage, storage_offset, nameAttr,
        fortran_attrs, data_attr, /*skip_rebox=*/aiir::UnitAttr{}, argNoAttr);
}

llvm::LogicalResult hlfir::DeclareOp::verify() {
  auto [hlfirVariableType, firVarType] = getDeclareOutputTypes(
      getMemref().getType(), hasExplicitLowerBounds(getShape()));
  if (firVarType != getResult(1).getType())
    return emitOpError("second result type must match input memref type, "
                       "unless it is a box with heap or pointer attribute");
  if (hlfirVariableType != getResult(0).getType())
    return emitOpError("first result type is inconsistent with variable "
                       "properties: expected ")
           << hlfirVariableType;
  if (getSkipRebox() && !llvm::isa<fir::BaseBoxType>(getMemref().getType()))
    return emitOpError(
        "skip_rebox attribute must only be set when the input is a box");
  // The rest of the argument verification is done by the
  // FortranVariableInterface verifier.
  auto fortranVar =
      aiir::cast<fir::FortranVariableOpInterface>(this->getOperation());
  return fortranVar.verifyDeclareLikeOpImpl(getMemref());
}

//===----------------------------------------------------------------------===//
// DesignateOp
//===----------------------------------------------------------------------===//

void hlfir::DesignateOp::build(
    aiir::OpBuilder &builder, aiir::OperationState &result,
    aiir::Type result_type, aiir::Value memref, llvm::StringRef component,
    aiir::Value component_shape, llvm::ArrayRef<Subscript> subscripts,
    aiir::ValueRange substring, std::optional<bool> complex_part,
    aiir::Value shape, aiir::ValueRange typeparams,
    fir::FortranVariableFlagsAttr fortran_attrs) {
  auto componentAttr =
      component.empty() ? aiir::StringAttr{} : builder.getStringAttr(component);
  llvm::SmallVector<aiir::Value> indices;
  llvm::SmallVector<bool> isTriplet;
  for (auto subscript : subscripts) {
    if (auto *triplet = std::get_if<Triplet>(&subscript)) {
      isTriplet.push_back(true);
      indices.push_back(std::get<0>(*triplet));
      indices.push_back(std::get<1>(*triplet));
      indices.push_back(std::get<2>(*triplet));
    } else {
      isTriplet.push_back(false);
      indices.push_back(std::get<aiir::Value>(subscript));
    }
  }
  auto isTripletAttr =
      aiir::DenseBoolArrayAttr::get(builder.getContext(), isTriplet);
  auto complexPartAttr =
      complex_part.has_value()
          ? aiir::BoolAttr::get(builder.getContext(), *complex_part)
          : aiir::BoolAttr{};
  build(builder, result, result_type, memref, componentAttr, component_shape,
        indices, isTripletAttr, substring, complexPartAttr, shape, typeparams,
        fortran_attrs);
}

void hlfir::DesignateOp::build(aiir::OpBuilder &builder,
                               aiir::OperationState &result,
                               aiir::Type result_type, aiir::Value memref,
                               aiir::ValueRange indices,
                               aiir::ValueRange typeparams,
                               fir::FortranVariableFlagsAttr fortran_attrs) {
  llvm::SmallVector<bool> isTriplet(indices.size(), false);
  auto isTripletAttr =
      aiir::DenseBoolArrayAttr::get(builder.getContext(), isTriplet);
  build(builder, result, result_type, memref,
        /*componentAttr=*/aiir::StringAttr{}, /*component_shape=*/aiir::Value{},
        indices, isTripletAttr, /*substring*/ aiir::ValueRange{},
        /*complexPartAttr=*/aiir::BoolAttr{}, /*shape=*/aiir::Value{},
        typeparams, fortran_attrs);
}

static aiir::ParseResult parseDesignatorIndices(
    aiir::OpAsmParser &parser,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &indices,
    aiir::DenseBoolArrayAttr &isTripletAttr) {
  llvm::SmallVector<bool> isTriplet;
  if (aiir::succeeded(parser.parseOptionalLParen())) {
    do {
      aiir::OpAsmParser::UnresolvedOperand i1, i2, i3;
      if (parser.parseOperand(i1))
        return aiir::failure();
      indices.push_back(i1);
      if (aiir::succeeded(parser.parseOptionalColon())) {
        if (parser.parseOperand(i2) || parser.parseColon() ||
            parser.parseOperand(i3))
          return aiir::failure();
        indices.push_back(i2);
        indices.push_back(i3);
        isTriplet.push_back(true);
      } else {
        isTriplet.push_back(false);
      }
    } while (aiir::succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return aiir::failure();
  }
  isTripletAttr = aiir::DenseBoolArrayAttr::get(parser.getContext(), isTriplet);
  return aiir::success();
}

static void
printDesignatorIndices(aiir::OpAsmPrinter &p, hlfir::DesignateOp designateOp,
                       aiir::OperandRange indices,
                       const aiir::DenseBoolArrayAttr &isTripletAttr) {
  if (!indices.empty()) {
    p << '(';
    unsigned i = 0;
    for (auto isTriplet : isTripletAttr.asArrayRef()) {
      if (isTriplet) {
        assert(i + 2 < indices.size() && "ill-formed indices");
        p << indices[i] << ":" << indices[i + 1] << ":" << indices[i + 2];
        i += 3;
      } else {
        p << indices[i++];
      }
      if (i != indices.size())
        p << ", ";
    }
    p << ')';
  }
}

static aiir::ParseResult
parseDesignatorComplexPart(aiir::OpAsmParser &parser,
                           aiir::BoolAttr &complexPart) {
  if (aiir::succeeded(parser.parseOptionalKeyword("imag")))
    complexPart = aiir::BoolAttr::get(parser.getContext(), true);
  else if (aiir::succeeded(parser.parseOptionalKeyword("real")))
    complexPart = aiir::BoolAttr::get(parser.getContext(), false);
  return aiir::success();
}

static void printDesignatorComplexPart(aiir::OpAsmPrinter &p,
                                       hlfir::DesignateOp designateOp,
                                       aiir::BoolAttr complexPartAttr) {
  if (complexPartAttr) {
    if (complexPartAttr.getValue())
      p << "imag";
    else
      p << "real";
  }
}
template <typename Op>
static llvm::LogicalResult verifyTypeparams(Op &op, aiir::Type elementType,
                                            unsigned numLenParam) {
  if (aiir::isa<fir::CharacterType>(elementType)) {
    if (numLenParam != 1)
      return op.emitOpError("must be provided one length parameter when the "
                            "result is a character");
  } else if (fir::isRecordWithTypeParameters(elementType)) {
    if (numLenParam !=
        aiir::cast<fir::RecordType>(elementType).getNumLenParams())
      return op.emitOpError("must be provided the same number of length "
                            "parameters as in the result derived type");
  } else if (numLenParam != 0) {
    return op.emitOpError(
        "must not be provided length parameters if the result "
        "type does not have length parameters");
  }
  return aiir::success();
}

llvm::LogicalResult hlfir::DesignateOp::verify() {
  aiir::Type memrefType = getMemref().getType();
  aiir::Type baseType = getFortranElementOrSequenceType(memrefType);
  aiir::Type baseElementType = fir::unwrapSequenceType(baseType);
  unsigned numSubscripts = getIsTriplet().size();
  unsigned subscriptsRank =
      llvm::count_if(getIsTriplet(), [](bool isTriplet) { return isTriplet; });
  unsigned outputRank = 0;
  aiir::Type outputElementType;
  bool hasBoxComponent;
  if (fir::useStrictVolatileVerification() &&
      fir::isa_volatile_type(memrefType) !=
          fir::isa_volatile_type(getResult().getType())) {
    return emitOpError("volatility mismatch between memref and result type")
           << " memref type: " << memrefType
           << " result type: " << getResult().getType();
  }
  if (getComponent()) {
    auto component = getComponent().value();
    auto recType = aiir::dyn_cast<fir::RecordType>(baseElementType);
    if (!recType)
      return emitOpError(
          "component must be provided only when the memref is a derived type");
    unsigned fieldIdx = recType.getFieldIndex(component);
    if (fieldIdx > recType.getNumFields()) {
      return emitOpError("component ")
             << component << " is not a component of memref element type "
             << recType;
    }
    aiir::Type fieldType = recType.getType(fieldIdx);
    aiir::Type componentBaseType = getFortranElementOrSequenceType(fieldType);
    hasBoxComponent = aiir::isa<fir::BaseBoxType>(fieldType);
    if (aiir::isa<fir::SequenceType>(componentBaseType) &&
        aiir::isa<fir::SequenceType>(baseType) &&
        (numSubscripts == 0 || subscriptsRank > 0))
      return emitOpError("indices must be provided and must not contain "
                         "triplets when both memref and component are arrays");
    if (numSubscripts != 0) {
      if (!aiir::isa<fir::SequenceType>(componentBaseType))
        return emitOpError("indices must not be provided if component appears "
                           "and is not an array component");
      if (!getComponentShape())
        return emitOpError(
            "component_shape must be provided when indexing a component");
      aiir::Type compShapeType = getComponentShape().getType();
      unsigned componentRank =
          aiir::cast<fir::SequenceType>(componentBaseType).getDimension();
      auto shapeType = aiir::dyn_cast<fir::ShapeType>(compShapeType);
      auto shapeShiftType = aiir::dyn_cast<fir::ShapeShiftType>(compShapeType);
      if (!((shapeType && shapeType.getRank() == componentRank) ||
            (shapeShiftType && shapeShiftType.getRank() == componentRank)))
        return emitOpError("component_shape must be a fir.shape or "
                           "fir.shapeshift with the rank of the component");
      if (numSubscripts > componentRank)
        return emitOpError("indices number must match array component rank");
    }
    if (auto baseSeqType = aiir::dyn_cast<fir::SequenceType>(baseType))
      // This case must come first to cover "array%array_comp(i, j)" that has
      // subscripts for the component but whose rank come from the base.
      outputRank = baseSeqType.getDimension();
    else if (numSubscripts != 0)
      outputRank = subscriptsRank;
    else if (auto componentSeqType =
                 aiir::dyn_cast<fir::SequenceType>(componentBaseType))
      outputRank = componentSeqType.getDimension();
    outputElementType = fir::unwrapSequenceType(componentBaseType);
  } else {
    outputElementType = baseElementType;
    unsigned baseTypeRank =
        aiir::isa<fir::SequenceType>(baseType)
            ? aiir::cast<fir::SequenceType>(baseType).getDimension()
            : 0;
    if (numSubscripts != 0) {
      if (baseTypeRank != numSubscripts)
        return emitOpError("indices number must match memref rank");
      outputRank = subscriptsRank;
    } else if (auto baseSeqType = aiir::dyn_cast<fir::SequenceType>(baseType)) {
      outputRank = baseSeqType.getDimension();
    }
  }

  if (!getSubstring().empty()) {
    if (!aiir::isa<fir::CharacterType>(outputElementType))
      return emitOpError("memref or component must have character type if "
                         "substring indices are provided");
    if (getSubstring().size() != 2)
      return emitOpError("substring must contain 2 indices when provided");
  }
  if (getComplexPart()) {
    if (auto cplx = aiir::dyn_cast<aiir::ComplexType>(outputElementType))
      outputElementType = cplx.getElementType();
    else
      return emitOpError("memref or component must have complex type if "
                         "complex_part is provided");
  }
  aiir::Type resultBaseType =
      getFortranElementOrSequenceType(getResult().getType());
  unsigned resultRank = 0;
  if (auto resultSeqType = aiir::dyn_cast<fir::SequenceType>(resultBaseType))
    resultRank = resultSeqType.getDimension();
  if (resultRank != outputRank)
    return emitOpError("result type rank is not consistent with operands, "
                       "expected rank ")
           << outputRank;
  aiir::Type resultElementType = fir::unwrapSequenceType(resultBaseType);
  // result type must match the one that was inferred here, except the character
  // length may differ because of substrings.
  if (resultElementType != outputElementType &&
      !(aiir::isa<fir::CharacterType>(resultElementType) &&
        aiir::isa<fir::CharacterType>(outputElementType)))
    return emitOpError(
               "result element type is not consistent with operands, expected ")
           << outputElementType;

  if (isBoxAddressType(getResult().getType())) {
    if (!hasBoxComponent || numSubscripts != 0 || !getSubstring().empty() ||
        getComplexPart())
      return emitOpError(
          "result type must only be a box address type if it designates a "
          "component that is a fir.box or fir.class and if there are no "
          "indices, substrings, and complex part");

  } else {
    if ((resultRank == 0) != !getShape())
      return emitOpError("shape must be provided if and only if the result is "
                         "an array that is not a box address");
    if (resultRank != 0) {
      auto shapeType = aiir::dyn_cast<fir::ShapeType>(getShape().getType());
      auto shapeShiftType =
          aiir::dyn_cast<fir::ShapeShiftType>(getShape().getType());
      if (!((shapeType && shapeType.getRank() == resultRank) ||
            (shapeShiftType && shapeShiftType.getRank() == resultRank)))
        return emitOpError("shape must be a fir.shape or fir.shapeshift with "
                           "the rank of the result");
    }
    if (auto res =
            verifyTypeparams(*this, outputElementType, getTypeparams().size());
        failed(res))
      return res;
  }
  return aiir::success();
}

std::optional<std::int64_t> hlfir::DesignateOp::getViewOffset(aiir::OpResult) {
  // TODO: we can compute the constant offset
  // based on the component/indices/etc.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ParentComponentOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::ParentComponentOp::verify() {
  aiir::Type baseType =
      hlfir::getFortranElementOrSequenceType(getMemref().getType());
  auto maybeInputSeqType = aiir::dyn_cast<fir::SequenceType>(baseType);
  unsigned inputTypeRank =
      maybeInputSeqType ? maybeInputSeqType.getDimension() : 0;
  unsigned shapeRank = 0;
  if (aiir::Value shape = getShape())
    if (auto shapeType = aiir::dyn_cast<fir::ShapeType>(shape.getType()))
      shapeRank = shapeType.getRank();
  if (inputTypeRank != shapeRank)
    return emitOpError(
        "must be provided a shape if and only if the base is an array");
  aiir::Type outputBaseType = hlfir::getFortranElementOrSequenceType(getType());
  auto maybeOutputSeqType = aiir::dyn_cast<fir::SequenceType>(outputBaseType);
  unsigned outputTypeRank =
      maybeOutputSeqType ? maybeOutputSeqType.getDimension() : 0;
  if (inputTypeRank != outputTypeRank)
    return emitOpError("result type rank must match input type rank");
  if (maybeOutputSeqType && maybeInputSeqType)
    for (auto [inputDim, outputDim] :
         llvm::zip(maybeInputSeqType.getShape(), maybeOutputSeqType.getShape()))
      if (inputDim != fir::SequenceType::getUnknownExtent() &&
          outputDim != fir::SequenceType::getUnknownExtent())
        if (inputDim != outputDim)
          return emitOpError(
              "result type extents are inconsistent with memref type");
  fir::RecordType baseRecType =
      aiir::dyn_cast<fir::RecordType>(hlfir::getFortranElementType(baseType));
  fir::RecordType outRecType = aiir::dyn_cast<fir::RecordType>(
      hlfir::getFortranElementType(outputBaseType));
  if (!baseRecType || !outRecType)
    return emitOpError("result type and input type must be derived types");

  // Note: result should not be a fir.class: its dynamic type is being set to
  // the parent type and allowing fir.class would break the operation codegen:
  // it would keep the input dynamic type.
  if (aiir::isa<fir::ClassType>(getType()))
    return emitOpError("result type must not be polymorphic");

  // The array results are known to not be dis-contiguous in most cases (the
  // exception being if the parent type was extended by a type without any
  // components): require a fir.box to be used for the result to carry the
  // strides.
  if (!aiir::isa<fir::BoxType>(getType()) &&
      (outputTypeRank != 0 || fir::isRecordWithTypeParameters(outRecType)))
    return emitOpError("result type must be a fir.box if the result is an "
                       "array or has length parameters");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// LogicalReductionOp
//===----------------------------------------------------------------------===//
template <typename LogicalReductionOp>
static llvm::LogicalResult
verifyLogicalReductionOp(LogicalReductionOp reductionOp) {
  aiir::Operation *op = reductionOp->getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  aiir::Value mask = reductionOp->getMask();
  aiir::Value dim = reductionOp->getDim();

  fir::SequenceType maskTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(mask.getType()));
  aiir::Type logicalTy = maskTy.getEleTy();
  llvm::ArrayRef<int64_t> maskShape = maskTy.getShape();

  aiir::Type resultType = results[0];
  if (aiir::isa<fir::LogicalType>(resultType)) {
    // Result is of the same type as MASK
    if ((resultType != logicalTy) && useStrictIntrinsicVerifier)
      return reductionOp->emitOpError(
          "result must have the same element type as MASK argument");

  } else if (auto resultExpr =
                 aiir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    // Result should only be in hlfir.expr form if it is an array
    if (maskShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return reductionOp->emitOpError("result must be an array");

      if ((resultExpr.getEleTy() != logicalTy) && useStrictIntrinsicVerifier)
        return reductionOp->emitOpError(
            "result must have the same element type as MASK argument");

      llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
      // Result has rank n-1
      if (resultShape.size() != (maskShape.size() - 1))
        return reductionOp->emitOpError(
            "result rank must be one less than MASK");
    } else {
      return reductionOp->emitOpError("result must be of logical type");
    }
  } else {
    return reductionOp->emitOpError("result must be of logical type");
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// AllOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::AllOp::verify() {
  return verifyLogicalReductionOp<hlfir::AllOp *>(this);
}

void hlfir::AllOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// AnyOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::AnyOp::verify() {
  return verifyLogicalReductionOp<hlfir::AnyOp *>(this);
}

void hlfir::AnyOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CountOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::CountOp::verify() {
  aiir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);
  aiir::Value mask = getMask();
  aiir::Value dim = getDim();

  fir::SequenceType maskTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(mask.getType()));
  llvm::ArrayRef<int64_t> maskShape = maskTy.getShape();

  aiir::Type resultType = results[0];
  if (auto resultExpr = aiir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    if (maskShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return emitOpError("result must be an array");

      llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
      // Result has rank n-1
      if (resultShape.size() != (maskShape.size() - 1))
        return emitOpError("result rank must be one less than MASK");
    } else {
      return emitOpError("result must be of numerical array type");
    }
  } else if (!hlfir::isFortranScalarNumericalType(resultType)) {
    return emitOpError("result must be of numerical scalar type");
  }

  return aiir::success();
}

void hlfir::CountOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getCharacterKind(aiir::Type t) {
  return aiir::cast<fir::CharacterType>(hlfir::getFortranElementType(t))
      .getFKind();
}

static std::optional<fir::CharacterType::LenType>
getCharacterLengthIfStatic(aiir::Type t) {
  if (auto charType =
          aiir::dyn_cast<fir::CharacterType>(hlfir::getFortranElementType(t)))
    if (charType.hasConstantLen())
      return charType.getLen();
  return std::nullopt;
}

llvm::LogicalResult hlfir::ConcatOp::verify() {
  if (getStrings().size() < 2)
    return emitOpError("must be provided at least two string operands");
  unsigned kind = getCharacterKind(getResult().getType());
  for (auto string : getStrings())
    if (kind != getCharacterKind(string.getType()))
      return emitOpError("strings must have the same KIND as the result type");
  return aiir::success();
}

void hlfir::ConcatOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result,
                            aiir::ValueRange strings, aiir::Value len) {
  fir::CharacterType::LenType resultTypeLen = 0;
  assert(!strings.empty() && "must contain operands");
  unsigned kind = getCharacterKind(strings[0].getType());
  for (auto string : strings)
    if (auto cstLen = getCharacterLengthIfStatic(string.getType())) {
      resultTypeLen += *cstLen;
    } else {
      resultTypeLen = fir::CharacterType::unknownLen();
      break;
    }
  auto resultType = hlfir::ExprType::get(
      builder.getContext(), hlfir::ExprType::Shape{},
      fir::CharacterType::get(builder.getContext(), kind, resultTypeLen),
      false);
  build(builder, result, resultType, strings, len);
}

void hlfir::ConcatOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CmpCharOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::CmpCharOp::verify() {
  aiir::Value lchr = getLchr();
  aiir::Value rchr = getRchr();

  unsigned kind = getCharacterKind(lchr.getType());
  if (kind != getCharacterKind(rchr.getType()))
    return emitOpError("character arguments must have the same KIND");

  switch (getPredicate()) {
  case aiir::arith::CmpIPredicate::slt:
  case aiir::arith::CmpIPredicate::sle:
  case aiir::arith::CmpIPredicate::eq:
  case aiir::arith::CmpIPredicate::ne:
  case aiir::arith::CmpIPredicate::sgt:
  case aiir::arith::CmpIPredicate::sge:
    break;
  default:
    return emitOpError("expected signed predicate");
  }

  return aiir::success();
}

void hlfir::CmpCharOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CharTrimOp
//===----------------------------------------------------------------------===//

void hlfir::CharTrimOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result, aiir::Value chr) {
  unsigned kind = getCharacterKind(chr.getType());
  auto resultType = hlfir::ExprType::get(
      builder.getContext(), hlfir::ExprType::Shape{},
      fir::CharacterType::get(builder.getContext(), kind,
                              fir::CharacterType::unknownLen()),
      /*polymorphic=*/false);
  build(builder, result, resultType, chr);
}

void hlfir::CharTrimOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::IndexOp::verify() {
  aiir::Value substr = getSubstr();
  aiir::Value str = getStr();

  unsigned charKind = getCharacterKind(substr.getType());
  if (charKind != getCharacterKind(str.getType()))
    return emitOpError("character arguments must have the same KIND");

  return aiir::success();
}

void hlfir::IndexOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// NumericalReductionOp
//===----------------------------------------------------------------------===//

template <typename NumericalReductionOp>
static llvm::LogicalResult
verifyArrayAndMaskForReductionOp(NumericalReductionOp reductionOp) {
  aiir::Value array = reductionOp->getArray();
  aiir::Value mask = reductionOp->getMask();

  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  if (mask) {
    fir::SequenceType maskSeq = aiir::dyn_cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(mask.getType()));
    llvm::ArrayRef<int64_t> maskShape;

    if (maskSeq)
      maskShape = maskSeq.getShape();

    if (!maskShape.empty()) {
      if (maskShape.size() != arrayShape.size())
        return reductionOp->emitWarning("MASK must be conformable to ARRAY");
      if (useStrictIntrinsicVerifier) {
        static_assert(fir::SequenceType::getUnknownExtent() ==
                      hlfir::ExprType::getUnknownExtent());
        constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
        for (std::size_t i = 0; i < arrayShape.size(); ++i) {
          int64_t arrayExtent = arrayShape[i];
          int64_t maskExtent = maskShape[i];
          if ((arrayExtent != maskExtent) && (arrayExtent != unknownExtent) &&
              (maskExtent != unknownExtent))
            return reductionOp->emitWarning(
                "MASK must be conformable to ARRAY");
        }
      }
    }
  }
  return aiir::success();
}

template <typename NumericalReductionOp>
static llvm::LogicalResult
verifyNumericalReductionOp(NumericalReductionOp reductionOp) {
  aiir::Operation *op = reductionOp->getOperation();
  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto res = verifyArrayAndMaskForReductionOp(reductionOp);
  if (failed(res))
    return res;

  aiir::Value array = reductionOp->getArray();
  aiir::Value dim = reductionOp->getDim();
  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  aiir::Type numTy = arrayTy.getEleTy();
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  aiir::Type resultType = results[0];
  if (hlfir::isFortranScalarNumericalType(resultType)) {
    // Result is of the same type as ARRAY
    if ((resultType != numTy) && useStrictIntrinsicVerifier)
      return reductionOp->emitOpError(
          "result must have the same element type as ARRAY argument");

  } else if (auto resultExpr =
                 aiir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    if (arrayShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return reductionOp->emitOpError("result must be an array");

      if ((resultExpr.getEleTy() != numTy) && useStrictIntrinsicVerifier)
        return reductionOp->emitOpError(
            "result must have the same element type as ARRAY argument");

      llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
      // Result has rank n-1
      if (resultShape.size() != (arrayShape.size() - 1))
        return reductionOp->emitOpError(
            "result rank must be one less than ARRAY");
    } else {
      return reductionOp->emitOpError(
          "result must be of numerical scalar type");
    }
  } else {
    return reductionOp->emitOpError("result must be of numerical scalar type");
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::ProductOp::verify() {
  return verifyNumericalReductionOp<hlfir::ProductOp *>(this);
}

void hlfir::ProductOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CharacterReductionOp
//===----------------------------------------------------------------------===//

template <typename CharacterReductionOp>
static llvm::LogicalResult
verifyCharacterReductionOp(CharacterReductionOp reductionOp) {
  aiir::Operation *op = reductionOp->getOperation();
  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto res = verifyArrayAndMaskForReductionOp(reductionOp);
  if (failed(res))
    return res;

  aiir::Value array = reductionOp->getArray();
  aiir::Value dim = reductionOp->getDim();
  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  aiir::Type numTy = arrayTy.getEleTy();
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  auto resultExpr = aiir::cast<hlfir::ExprType>(results[0]);
  aiir::Type resultType = resultExpr.getEleTy();
  assert(aiir::isa<fir::CharacterType>(resultType) &&
         "result must be character");

  // Result is of the same type as ARRAY
  if ((resultType != numTy) && useStrictIntrinsicVerifier)
    return reductionOp->emitOpError(
        "result must have the same element type as ARRAY argument");

  if (arrayShape.size() > 1 && dim != nullptr) {
    if (!resultExpr.isArray())
      return reductionOp->emitOpError("result must be an array");
    llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
    // Result has rank n-1
    if (resultShape.size() != (arrayShape.size() - 1))
      return reductionOp->emitOpError(
          "result rank must be one less than ARRAY");
  } else if (!resultExpr.isScalar()) {
    return reductionOp->emitOpError("result must be scalar character");
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// MaxvalOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::MaxvalOp::verify() {
  aiir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto resultExpr = aiir::dyn_cast<hlfir::ExprType>(results[0]);
  if (resultExpr && aiir::isa<fir::CharacterType>(resultExpr.getEleTy())) {
    return verifyCharacterReductionOp<hlfir::MaxvalOp *>(this);
  }
  return verifyNumericalReductionOp<hlfir::MaxvalOp *>(this);
}

void hlfir::MaxvalOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MinvalOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::MinvalOp::verify() {
  aiir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto resultExpr = aiir::dyn_cast<hlfir::ExprType>(results[0]);
  if (resultExpr && aiir::isa<fir::CharacterType>(resultExpr.getEleTy())) {
    return verifyCharacterReductionOp<hlfir::MinvalOp *>(this);
  }
  return verifyNumericalReductionOp<hlfir::MinvalOp *>(this);
}

void hlfir::MinvalOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MinlocOp
//===----------------------------------------------------------------------===//

template <typename NumericalReductionOp>
static llvm::LogicalResult
verifyResultForMinMaxLoc(NumericalReductionOp reductionOp) {
  aiir::Operation *op = reductionOp->getOperation();
  auto results = op->getResultTypes();
  assert(results.size() == 1);

  aiir::Value array = reductionOp->getArray();
  aiir::Value dim = reductionOp->getDim();
  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  aiir::Type resultType = results[0];
  if (dim && arrayShape.size() == 1) {
    if (!fir::isa_integer(resultType))
      return reductionOp->emitOpError("result must be scalar integer");
  } else if (auto resultExpr =
                 aiir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    if (!resultExpr.isArray())
      return reductionOp->emitOpError("result must be an array");

    if (!fir::isa_integer(resultExpr.getEleTy()))
      return reductionOp->emitOpError("result must have integer elements");

    llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
    // With dim the result has rank n-1
    if (dim && resultShape.size() != (arrayShape.size() - 1))
      return reductionOp->emitOpError(
          "result rank must be one less than ARRAY");
    // With dim the result has rank n
    if (!dim && resultShape.size() != 1)
      return reductionOp->emitOpError("result rank must be 1");
  } else {
    return reductionOp->emitOpError("result must be of numerical expr type");
  }
  return aiir::success();
}

llvm::LogicalResult hlfir::MinlocOp::verify() {
  auto res = verifyArrayAndMaskForReductionOp(this);
  if (failed(res))
    return res;

  return verifyResultForMinMaxLoc(this);
}

void hlfir::MinlocOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MaxlocOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::MaxlocOp::verify() {
  auto res = verifyArrayAndMaskForReductionOp(this);
  if (failed(res))
    return res;

  return verifyResultForMinMaxLoc(this);
}

void hlfir::MaxlocOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// SetLengthOp
//===----------------------------------------------------------------------===//

void hlfir::SetLengthOp::build(aiir::OpBuilder &builder,
                               aiir::OperationState &result, aiir::Value string,
                               aiir::Value len) {
  fir::CharacterType::LenType resultTypeLen = fir::CharacterType::unknownLen();
  if (auto cstLen = fir::getIntIfConstant(len))
    resultTypeLen = *cstLen;
  unsigned kind = getCharacterKind(string.getType());
  auto resultType = hlfir::ExprType::get(
      builder.getContext(), hlfir::ExprType::Shape{},
      fir::CharacterType::get(builder.getContext(), kind, resultTypeLen),
      false);
  build(builder, result, resultType, string, len);
}

void hlfir::SetLengthOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::SumOp::verify() {
  return verifyNumericalReductionOp<hlfir::SumOp *>(this);
}

void hlfir::SumOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// DotProductOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::DotProductOp::verify() {
  aiir::Value lhs = getLhs();
  aiir::Value rhs = getRhs();
  fir::SequenceType lhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(lhs.getType()));
  fir::SequenceType rhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(rhs.getType()));
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  aiir::Type lhsEleTy = lhsTy.getEleTy();
  aiir::Type rhsEleTy = rhsTy.getEleTy();
  aiir::Type resultTy = getResult().getType();

  if ((lhsRank != 1) || (rhsRank != 1))
    return emitOpError("both arrays must have rank 1");

  int64_t lhsSize = lhsShape[0];
  int64_t rhsSize = rhsShape[0];

  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if ((lhsSize != unknownExtent) && (rhsSize != unknownExtent) &&
      (lhsSize != rhsSize) && useStrictIntrinsicVerifier)
    return emitOpError("both arrays must have the same size");

  if (useStrictIntrinsicVerifier) {
    if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
        aiir::isa<fir::LogicalType>(rhsEleTy))
      return emitOpError("if one array is logical, so should the other be");

    if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
        aiir::isa<fir::LogicalType>(resultTy))
      return emitOpError("the result type should be a logical only if the "
                         "argument types are logical");
  }

  if (!hlfir::isFortranScalarNumericalType(resultTy) &&
      !aiir::isa<fir::LogicalType>(resultTy))
    return emitOpError(
        "the result must be of scalar numerical or logical type");

  return aiir::success();
}

void hlfir::DotProductOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::MatmulOp::verify() {
  aiir::Value lhs = getLhs();
  aiir::Value rhs = getRhs();
  fir::SequenceType lhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(lhs.getType()));
  fir::SequenceType rhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(rhs.getType()));
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  aiir::Type lhsEleTy = lhsTy.getEleTy();
  aiir::Type rhsEleTy = rhsTy.getEleTy();
  hlfir::ExprType resultTy = aiir::cast<hlfir::ExprType>(getResult().getType());
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  aiir::Type resultEleTy = resultTy.getEleTy();

  if (((lhsRank != 1) && (lhsRank != 2)) || ((rhsRank != 1) && (rhsRank != 2)))
    return emitOpError("array must have either rank 1 or rank 2");

  if ((lhsRank == 1) && (rhsRank == 1))
    return emitOpError("at least one array must have rank 2");

  if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
      aiir::isa<fir::LogicalType>(rhsEleTy))
    return emitOpError("if one array is logical, so should the other be");

  if (!useStrictIntrinsicVerifier)
    return aiir::success();

  int64_t lastLhsDim = lhsShape[lhsRank - 1];
  int64_t firstRhsDim = rhsShape[0];
  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if (lastLhsDim != firstRhsDim)
    if ((lastLhsDim != unknownExtent) && (firstRhsDim != unknownExtent))
      return emitOpError(
          "the last dimension of LHS should match the first dimension of RHS");

  if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
      aiir::isa<fir::LogicalType>(resultEleTy))
    return emitOpError("the result type should be a logical only if the "
                       "argument types are logical");

  llvm::SmallVector<int64_t, 2> expectedResultShape;
  if (lhsRank == 2) {
    if (rhsRank == 2) {
      expectedResultShape.push_back(lhsShape[0]);
      expectedResultShape.push_back(rhsShape[1]);
    } else {
      // rhsRank == 1
      expectedResultShape.push_back(lhsShape[0]);
    }
  } else {
    // lhsRank == 1
    // rhsRank == 2
    expectedResultShape.push_back(rhsShape[1]);
  }
  if (resultShape.size() != expectedResultShape.size())
    return emitOpError("incorrect result shape");
  if (resultShape[0] != expectedResultShape[0] &&
      expectedResultShape[0] != unknownExtent)
    return emitOpError("incorrect result shape");
  if (resultShape.size() == 2 && resultShape[1] != expectedResultShape[1] &&
      expectedResultShape[1] != unknownExtent)
    return emitOpError("incorrect result shape");

  return aiir::success();
}

llvm::LogicalResult
hlfir::MatmulOp::canonicalize(MatmulOp matmulOp,
                              aiir::PatternRewriter &rewriter) {
  // the only two uses of the transposed matrix should be for the hlfir.matmul
  // and hlfir.destroy
  auto isOtherwiseUnused = [&](hlfir::TransposeOp transposeOp) -> bool {
    std::size_t numUses = 0;
    for (aiir::Operation *user : transposeOp.getResult().getUsers()) {
      ++numUses;
      if (user == matmulOp)
        continue;
      if (aiir::dyn_cast_or_null<hlfir::DestroyOp>(user))
        continue;
      // some other use!
      return false;
    }
    return numUses <= 2;
  };

  aiir::Value lhs = matmulOp.getLhs();
  // Rewrite MATMUL(TRANSPOSE(lhs), rhs) => hlfir.matmul_transpose lhs, rhs
  if (auto transposeOp = lhs.getDefiningOp<hlfir::TransposeOp>()) {
    if (isOtherwiseUnused(transposeOp)) {
      aiir::Location loc = matmulOp.getLoc();
      aiir::Type resultTy = matmulOp.getResult().getType();
      auto matmulTransposeOp = hlfir::MatmulTransposeOp::create(
          rewriter, loc, resultTy, transposeOp.getArray(), matmulOp.getRhs(),
          matmulOp.getFastmathAttr());

      // we don't need to remove any hlfir.destroy because it will be needed for
      // the new intrinsic result anyway
      rewriter.replaceOp(matmulOp, matmulTransposeOp.getResult());

      // but we do need to get rid of the hlfir.destroy for the hlfir.transpose
      // result (which is entirely removed)
      llvm::SmallVector<aiir::Operation *> users(
          transposeOp->getResult(0).getUsers());
      for (aiir::Operation *user : users)
        if (auto destroyOp = aiir::dyn_cast_or_null<hlfir::DestroyOp>(user))
          rewriter.eraseOp(destroyOp);
      rewriter.eraseOp(transposeOp);

      return aiir::success();
    }
  }

  return aiir::failure();
}

void hlfir::MatmulOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::TransposeOp::verify() {
  aiir::Value array = getArray();
  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  llvm::ArrayRef<int64_t> inShape = arrayTy.getShape();
  std::size_t rank = inShape.size();
  aiir::Type eleTy = arrayTy.getEleTy();
  hlfir::ExprType resultTy = aiir::cast<hlfir::ExprType>(getResult().getType());
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  std::size_t resultRank = resultShape.size();
  aiir::Type resultEleTy = resultTy.getEleTy();

  if (rank != 2 || resultRank != 2)
    return emitOpError("input and output arrays should have rank 2");

  if (!useStrictIntrinsicVerifier)
    return aiir::success();

  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if ((inShape[0] != resultShape[1]) && (inShape[0] != unknownExtent))
    return emitOpError("output shape does not match input array");
  if ((inShape[1] != resultShape[0]) && (inShape[1] != unknownExtent))
    return emitOpError("output shape does not match input array");

  if (eleTy != resultEleTy)
    return emitOpError(
        "input and output arrays should have the same element type");

  return aiir::success();
}

void hlfir::TransposeOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MatmulTransposeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::MatmulTransposeOp::verify() {
  aiir::Value lhs = getLhs();
  aiir::Value rhs = getRhs();
  fir::SequenceType lhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(lhs.getType()));
  fir::SequenceType rhsTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(rhs.getType()));
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  aiir::Type lhsEleTy = lhsTy.getEleTy();
  aiir::Type rhsEleTy = rhsTy.getEleTy();
  hlfir::ExprType resultTy = aiir::cast<hlfir::ExprType>(getResult().getType());
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  aiir::Type resultEleTy = resultTy.getEleTy();

  // lhs must have rank 2 for the transpose to be valid
  if ((lhsRank != 2) || ((rhsRank != 1) && (rhsRank != 2)))
    return emitOpError("array must have either rank 1 or rank 2");

  if (!useStrictIntrinsicVerifier)
    return aiir::success();

  if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
      aiir::isa<fir::LogicalType>(rhsEleTy))
    return emitOpError("if one array is logical, so should the other be");

  // for matmul we compare the last dimension of lhs with the first dimension of
  // rhs, but for MatmulTranspose, dimensions of lhs are inverted by the
  // transpose
  int64_t firstLhsDim = lhsShape[0];
  int64_t firstRhsDim = rhsShape[0];
  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if (firstLhsDim != firstRhsDim)
    if ((firstLhsDim != unknownExtent) && (firstRhsDim != unknownExtent))
      return emitOpError(
          "the first dimension of LHS should match the first dimension of RHS");

  if (aiir::isa<fir::LogicalType>(lhsEleTy) !=
      aiir::isa<fir::LogicalType>(resultEleTy))
    return emitOpError("the result type should be a logical only if the "
                       "argument types are logical");

  llvm::SmallVector<int64_t, 2> expectedResultShape;
  if (rhsRank == 2) {
    expectedResultShape.push_back(lhsShape[1]);
    expectedResultShape.push_back(rhsShape[1]);
  } else {
    // rhsRank == 1
    expectedResultShape.push_back(lhsShape[1]);
  }
  if (resultShape.size() != expectedResultShape.size())
    return emitOpError("incorrect result shape");
  if (resultShape[0] != expectedResultShape[0])
    return emitOpError("incorrect result shape");
  if (resultShape.size() == 2 && resultShape[1] != expectedResultShape[1])
    return emitOpError("incorrect result shape");

  return aiir::success();
}

void hlfir::MatmulTransposeOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// Array shifts: CShiftOp/EOShiftOp
//===----------------------------------------------------------------------===//

template <typename Op>
static llvm::LogicalResult verifyArrayShift(Op op) {
  aiir::Value array = op.getArray();
  fir::SequenceType arrayTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  llvm::ArrayRef<int64_t> inShape = arrayTy.getShape();
  std::size_t arrayRank = inShape.size();
  aiir::Type eleTy = arrayTy.getEleTy();
  hlfir::ExprType resultTy =
      aiir::cast<hlfir::ExprType>(op.getResult().getType());
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  std::size_t resultRank = resultShape.size();
  aiir::Type resultEleTy = resultTy.getEleTy();
  aiir::Value shift = op.getShift();
  aiir::Type shiftTy = hlfir::getFortranElementOrSequenceType(shift.getType());

  if (auto match = areMatchingTypes(
          op, eleTy, resultEleTy,
          /*allowCharacterLenMismatch=*/!useStrictIntrinsicVerifier);
      match.failed())
    return op.emitOpError(
        "input and output arrays should have the same element type");

  if (arrayRank != resultRank)
    return op.emitOpError("input and output arrays should have the same rank");

  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  for (auto [inDim, resultDim] : llvm::zip(inShape, resultShape))
    if (inDim != unknownExtent && resultDim != unknownExtent &&
        inDim != resultDim)
      return op.emitOpError(
          "output array's shape conflicts with the input array's shape");

  int64_t dimVal = -1;
  if (!op.getDim())
    dimVal = 1;
  else if (auto dim = fir::getIntIfConstant(op.getDim()))
    dimVal = *dim;

  // The DIM argument may be statically invalid (e.g. exceed the
  // input array rank) in dead code after constant propagation,
  // so avoid some checks unless useStrictIntrinsicVerifier is true.
  if (useStrictIntrinsicVerifier && dimVal != -1) {
    if (dimVal < 1)
      return op.emitOpError("DIM must be >= 1");
    if (dimVal > static_cast<int64_t>(arrayRank))
      return op.emitOpError("DIM must be <= input array's rank");
  }

  // A helper lambda to verify the shape of the array types of
  // certain operands of the array shift (e.g. the SHIFT and BOUNDARY operands).
  auto verifyOperandTypeShape = [&](aiir::Type type,
                                    llvm::Twine name) -> llvm::LogicalResult {
    if (auto opndSeqTy = aiir::dyn_cast<fir::SequenceType>(type)) {
      // The operand is an array. Verify the rank and the shape (if DIM is
      // constant).
      llvm::ArrayRef<int64_t> opndShape = opndSeqTy.getShape();
      std::size_t opndRank = opndShape.size();
      if (opndRank != arrayRank - 1)
        return op.emitOpError(
            name + "'s rank must be 1 less than the input array's rank");

      if (useStrictIntrinsicVerifier && dimVal != -1) {
        // The operand's shape must be
        // [d(1), d(2), ..., d(DIM-1), d(DIM+1), ..., d(n)],
        // where [d(1), d(2), ..., d(n)] is the shape of the ARRAY.
        int64_t arrayDimIdx = 0;
        int64_t opndDimIdx = 0;
        for (auto opndDim : opndShape) {
          if (arrayDimIdx == dimVal - 1)
            ++arrayDimIdx;

          if (inShape[arrayDimIdx] != unknownExtent &&
              opndDim != unknownExtent && inShape[arrayDimIdx] != opndDim)
            return op.emitOpError("SHAPE(ARRAY)(" +
                                  llvm::Twine(arrayDimIdx + 1) +
                                  ") must be equal to SHAPE(" + name + ")(" +
                                  llvm::Twine(opndDimIdx + 1) +
                                  "): " + llvm::Twine(inShape[arrayDimIdx]) +
                                  " != " + llvm::Twine(opndDim));
          ++arrayDimIdx;
          ++opndDimIdx;
        }
      }
    }
    return aiir::success();
  };

  if (failed(verifyOperandTypeShape(shiftTy, "SHIFT")))
    return aiir::failure();

  if constexpr (std::is_same_v<Op, hlfir::EOShiftOp>) {
    if (aiir::Value boundary = op.getBoundary()) {
      aiir::Type boundaryTy =
          hlfir::getFortranElementOrSequenceType(boundary.getType());
      // In case of polymorphic ARRAY type, the BOUNDARY's element type
      // may not match the ARRAY's element type.
      if (!hlfir::isPolymorphicType(array.getType()))
        if (auto match = areMatchingTypes(
                op, eleTy, hlfir::getFortranElementType(boundaryTy),
                /*allowCharacterLenMismatch=*/!useStrictIntrinsicVerifier);
            match.failed())
          return op.emitOpError(
              "ARRAY and BOUNDARY operands must have the same element type");
      if (failed(verifyOperandTypeShape(boundaryTy, "BOUNDARY")))
        return aiir::failure();
    }
  }

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CShiftOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::CShiftOp::verify() {
  return verifyArrayShift(*this);
}

void hlfir::CShiftOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// EOShiftOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::EOShiftOp::verify() {
  return verifyArrayShift(*this);
}

void hlfir::EOShiftOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::ReshapeOp::verify() {
  auto results = getOperation()->getResultTypes();
  assert(results.size() == 1);
  hlfir::ExprType resultType = aiir::cast<hlfir::ExprType>(results[0]);
  aiir::Value array = getArray();
  auto arrayType = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(array.getType()));
  if (auto match = areMatchingTypes(
          *this, hlfir::getFortranElementType(resultType),
          arrayType.getElementType(),
          /*allowCharacterLenMismatch=*/!useStrictIntrinsicVerifier);
      match.failed())
    return emitOpError("ARRAY and the result must have the same element type");
  if (hlfir::isPolymorphicType(resultType) !=
      hlfir::isPolymorphicType(array.getType()))
    return emitOpError("ARRAY must be polymorphic iff result is polymorphic");

  aiir::Value shape = getShape();
  auto shapeArrayType = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(shape.getType()));
  if (shapeArrayType.getDimension() != 1)
    return emitOpError("SHAPE must be an array of rank 1");
  if (!aiir::isa<aiir::IntegerType>(shapeArrayType.getElementType()))
    return emitOpError("SHAPE must be an integer array");
  if (shapeArrayType.hasDynamicExtents())
    return emitOpError("SHAPE must have known size");
  if (shapeArrayType.getConstantArraySize() != resultType.getRank())
    return emitOpError("SHAPE's extent must match the result rank");

  if (aiir::Value pad = getPad()) {
    auto padArrayType = aiir::cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(pad.getType()));
    if (auto match = areMatchingTypes(
            *this, arrayType.getElementType(), padArrayType.getElementType(),
            /*allowCharacterLenMismatch=*/!useStrictIntrinsicVerifier);
        match.failed())
      return emitOpError("ARRAY and PAD must be of the same type");
  }

  if (aiir::Value order = getOrder()) {
    auto orderArrayType = aiir::cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(order.getType()));
    if (orderArrayType.getDimension() != 1)
      return emitOpError("ORDER must be an array of rank 1");
    if (!aiir::isa<aiir::IntegerType>(orderArrayType.getElementType()))
      return emitOpError("ORDER must be an integer array");
  }

  return aiir::success();
}

void hlfir::ReshapeOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// AssociateOp
//===----------------------------------------------------------------------===//

void hlfir::AssociateOp::build(aiir::OpBuilder &builder,
                               aiir::OperationState &result, aiir::Value source,
                               llvm::StringRef uniq_name, aiir::Value shape,
                               aiir::ValueRange typeparams,
                               fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  aiir::Type dataType = getFortranElementOrSequenceType(source.getType());

  // Preserve polymorphism of polymorphic expr.
  aiir::Type firVarType;
  auto sourceExprType = aiir::dyn_cast<hlfir::ExprType>(source.getType());
  if (sourceExprType && sourceExprType.isPolymorphic())
    firVarType = fir::ClassType::get(dataType);
  else
    firVarType = fir::ReferenceType::get(dataType);

  aiir::Type hlfirVariableType =
      DeclareOp::getHLFIRVariableType(firVarType, /*hasExplicitLbs=*/false);
  aiir::Type i1Type = builder.getI1Type();
  build(builder, result, {hlfirVariableType, firVarType, i1Type}, source, shape,
        typeparams, nameAttr, fortran_attrs);
}

void hlfir::AssociateOp::build(
    aiir::OpBuilder &builder, aiir::OperationState &result, aiir::Value source,
    aiir::Value shape, aiir::ValueRange typeparams,
    fir::FortranVariableFlagsAttr fortran_attrs,
    llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  aiir::Type dataType = getFortranElementOrSequenceType(source.getType());

  // Preserve polymorphism of polymorphic expr.
  aiir::Type firVarType;
  auto sourceExprType = aiir::dyn_cast<hlfir::ExprType>(source.getType());
  if (sourceExprType && sourceExprType.isPolymorphic())
    firVarType = fir::ClassType::get(dataType);
  else
    firVarType = fir::ReferenceType::get(dataType);

  aiir::Type hlfirVariableType =
      DeclareOp::getHLFIRVariableType(firVarType, /*hasExplicitLbs=*/false);
  aiir::Type i1Type = builder.getI1Type();
  build(builder, result, {hlfirVariableType, firVarType, i1Type}, source, shape,
        typeparams, {}, fortran_attrs);
  result.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// EndAssociateOp
//===----------------------------------------------------------------------===//

void hlfir::EndAssociateOp::build(aiir::OpBuilder &builder,
                                  aiir::OperationState &result,
                                  hlfir::AssociateOp associate) {
  aiir::Value hlfirBase = associate.getBase();
  aiir::Value firBase = associate.getFirBase();
  // If EndAssociateOp may need to initiate the deallocation
  // of allocatable components, it has to have access to the variable
  // definition, so we cannot use the FIR base as the operand.
  return build(builder, result,
               hlfir::mayHaveAllocatableComponent(hlfirBase.getType())
                   ? hlfirBase
                   : firBase,
               associate.getMustFreeStrorageFlag());
}

llvm::LogicalResult hlfir::EndAssociateOp::verify() {
  aiir::Value var = getVar();
  if (hlfir::mayHaveAllocatableComponent(var.getType()) &&
      !hlfir::isFortranEntity(var))
    return emitOpError("that requires components deallocation must have var "
                       "operand that is a Fortran entity");

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// AsExprOp
//===----------------------------------------------------------------------===//

void hlfir::AsExprOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result, aiir::Value var,
                            aiir::Value mustFree) {
  aiir::Type resultType = hlfir::getExprType(var.getType());
  return build(builder, result, resultType, var, mustFree);
}

void hlfir::AsExprOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  // this isn't a transformational intrinsic but follows the same pattern: it
  // creates a hlfir.expr and so needs to have an allocation effect, plus it
  // might have a pointer-like argument, in which case it has a read effect
  // upon those
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// ElementalOp
//===----------------------------------------------------------------------===//

/// Common builder for ElementalOp and ElementalAddrOp to add the arguments and
/// create the elemental body. Result and clean-up body must be handled in
/// specific builders.
template <typename Op>
static void buildElemental(aiir::OpBuilder &builder,
                           aiir::OperationState &odsState, aiir::Value shape,
                           aiir::Value mold, aiir::ValueRange typeparams,
                           bool isUnordered) {
  odsState.addOperands(shape);
  if (mold)
    odsState.addOperands(mold);
  odsState.addOperands(typeparams);
  odsState.addAttribute(
      Op::getOperandSegmentSizesAttrName(odsState.name),
      builder.getDenseI32ArrayAttr({/*shape=*/1, (mold ? 1 : 0),
                                    static_cast<int32_t>(typeparams.size())}));
  if (isUnordered)
    odsState.addAttribute(Op::getUnorderedAttrName(odsState.name),
                          isUnordered ? builder.getUnitAttr() : nullptr);
  aiir::Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new aiir::Block{});
  if (auto shapeType = aiir::dyn_cast<fir::ShapeType>(shape.getType())) {
    unsigned dim = shapeType.getRank();
    aiir::Type indexType = builder.getIndexType();
    for (unsigned d = 0; d < dim; ++d)
      bodyRegion->front().addArgument(indexType, odsState.location);
  }
}

void hlfir::ElementalOp::build(aiir::OpBuilder &builder,
                               aiir::OperationState &odsState,
                               aiir::Type resultType, aiir::Value shape,
                               aiir::Value mold, aiir::ValueRange typeparams,
                               bool isUnordered) {
  odsState.addTypes(resultType);
  buildElemental<hlfir::ElementalOp>(builder, odsState, shape, mold, typeparams,
                                     isUnordered);
}

aiir::Value hlfir::ElementalOp::getElementEntity() {
  return aiir::cast<hlfir::YieldElementOp>(getBody()->back()).getElementValue();
}

llvm::LogicalResult hlfir::ElementalOp::verify() {
  aiir::Value mold = getMold();
  hlfir::ExprType resultType = aiir::cast<hlfir::ExprType>(getType());
  if (!!mold != resultType.isPolymorphic())
    return emitOpError("result must be polymorphic when mold is present "
                       "and vice versa");

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

void hlfir::ApplyOp::build(aiir::OpBuilder &builder,
                           aiir::OperationState &odsState, aiir::Value expr,
                           aiir::ValueRange indices,
                           aiir::ValueRange typeparams) {
  aiir::Type resultType = expr.getType();
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(resultType))
    resultType = exprType.getElementExprType();
  build(builder, odsState, resultType, expr, indices, typeparams);
}

//===----------------------------------------------------------------------===//
// NullOp
//===----------------------------------------------------------------------===//

void hlfir::NullOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &odsState) {
  return build(builder, odsState,
               fir::ReferenceType::get(builder.getNoneType()));
}

//===----------------------------------------------------------------------===//
// DestroyOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::DestroyOp::verify() {
  if (mustFinalizeExpr()) {
    aiir::Value expr = getExpr();
    hlfir::ExprType exprTy = aiir::cast<hlfir::ExprType>(expr.getType());
    aiir::Type elemTy = hlfir::getFortranElementType(exprTy);
    if (!aiir::isa<fir::RecordType>(elemTy))
      return emitOpError(
          "the element type must be finalizable, when 'finalize' is set");
  }

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CopyInOp
//===----------------------------------------------------------------------===//

void hlfir::CopyInOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &odsState, aiir::Value var,
                            aiir::Value tempBox, aiir::Value var_is_present) {
  return build(builder, odsState, {var.getType(), builder.getI1Type()}, var,
               tempBox, var_is_present);
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

void hlfir::ShapeOfOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result, aiir::Value expr) {
  hlfir::ExprType exprTy = aiir::cast<hlfir::ExprType>(expr.getType());
  aiir::Type type = fir::ShapeType::get(builder.getContext(), exprTy.getRank());
  build(builder, result, type, expr);
}

std::size_t hlfir::ShapeOfOp::getRank() {
  aiir::Type resTy = getResult().getType();
  fir::ShapeType shape = aiir::cast<fir::ShapeType>(resTy);
  return shape.getRank();
}

llvm::LogicalResult hlfir::ShapeOfOp::verify() {
  aiir::Value expr = getExpr();
  hlfir::ExprType exprTy = aiir::cast<hlfir::ExprType>(expr.getType());
  std::size_t exprRank = exprTy.getShape().size();

  if (exprRank == 0)
    return emitOpError("cannot get the shape of a shape-less expression");

  std::size_t shapeRank = getRank();
  if (shapeRank != exprRank)
    return emitOpError("result rank and expr rank do not match");

  return aiir::success();
}

llvm::LogicalResult
hlfir::ShapeOfOp::canonicalize(ShapeOfOp shapeOf,
                               aiir::PatternRewriter &rewriter) {
  // if extent information is available at compile time, immediately fold the
  // hlfir.shape_of into a fir.shape
  aiir::Location loc = shapeOf.getLoc();
  hlfir::ExprType expr =
      aiir::cast<hlfir::ExprType>(shapeOf.getExpr().getType());

  aiir::Value shape = hlfir::genExprShape(rewriter, loc, expr);
  if (!shape)
    // shape information is not available at compile time
    return llvm::LogicalResult::failure();

  rewriter.replaceOp(shapeOf, shape);
  return llvm::LogicalResult::success();
}

aiir::OpFoldResult hlfir::ShapeOfOp::fold(FoldAdaptor adaptor) {
  if (matchPattern(getExpr(), aiir::m_Op<hlfir::ElementalOp>())) {
    auto elementalOp =
        aiir::cast<hlfir::ElementalOp>(getExpr().getDefiningOp());
    return elementalOp.getShape();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// GetExtent
//===----------------------------------------------------------------------===//

void hlfir::GetExtentOp::build(aiir::OpBuilder &builder,
                               aiir::OperationState &result, aiir::Value shape,
                               unsigned dim) {
  aiir::Type indexTy = builder.getIndexType();
  aiir::IntegerAttr dimAttr = aiir::IntegerAttr::get(indexTy, dim);
  build(builder, result, indexTy, shape, dimAttr);
}

llvm::LogicalResult hlfir::GetExtentOp::verify() {
  fir::ShapeType shapeTy = aiir::cast<fir::ShapeType>(getShape().getType());
  std::uint64_t rank = shapeTy.getRank();
  llvm::APInt dim = getDim();
  if (dim.sge(rank))
    return emitOpError("dimension index out of bounds");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// RegionAssignOp
//===----------------------------------------------------------------------===//

/// Add a fir.end terminator to a parsed region if it does not already has a
/// terminator.
static void ensureTerminator(aiir::Region &region, aiir::Builder &builder,
                             aiir::Location loc) {
  // Borrow YielOp::ensureTerminator AIIR generated implementation to add a
  // fir.end if there is no terminator. This has nothing to do with YielOp,
  // other than the fact that yieldOp has the
  // SingleBlocklicitTerminator<"fir::FirEndOp"> interface that
  // cannot be added on other HLFIR operations with several regions which are
  // not all terminated the same way.
  hlfir::YieldOp::ensureTerminator(region, builder, loc);
}

aiir::ParseResult hlfir::RegionAssignOp::parse(aiir::OpAsmParser &parser,
                                               aiir::OperationState &result) {
  aiir::Region &rhsRegion = *result.addRegion();
  if (parser.parseRegion(rhsRegion))
    return aiir::failure();
  aiir::Region &lhsRegion = *result.addRegion();
  if (parser.parseKeyword("to") || parser.parseRegion(lhsRegion))
    return aiir::failure();
  aiir::Region &userDefinedAssignmentRegion = *result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("user_defined_assign"))) {
    aiir::OpAsmParser::Argument rhsArg, lhsArg;
    if (parser.parseLParen() || parser.parseArgument(rhsArg) ||
        parser.parseColon() || parser.parseType(rhsArg.type) ||
        parser.parseRParen() || parser.parseKeyword("to") ||
        parser.parseLParen() || parser.parseArgument(lhsArg) ||
        parser.parseColon() || parser.parseType(lhsArg.type) ||
        parser.parseRParen())
      return aiir::failure();
    if (parser.parseRegion(userDefinedAssignmentRegion, {rhsArg, lhsArg}))
      return aiir::failure();
    ensureTerminator(userDefinedAssignmentRegion, parser.getBuilder(),
                     result.location);
  }
  return aiir::success();
}

void hlfir::RegionAssignOp::print(aiir::OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRhsRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p << " to ";
  p.printRegion(getLhsRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (!getUserDefinedAssignment().empty()) {
    p << " user_defined_assign ";
    aiir::Value userAssignmentRhs = getUserAssignmentRhs();
    aiir::Value userAssignmentLhs = getUserAssignmentLhs();
    p << " (" << userAssignmentRhs << ": " << userAssignmentRhs.getType()
      << ") to (";
    p << userAssignmentLhs << ": " << userAssignmentLhs.getType() << ") ";
    p.printRegion(getUserDefinedAssignment(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

static aiir::Operation *getTerminator(aiir::Region &region) {
  if (region.empty() || region.back().empty())
    return nullptr;
  return &region.back().back();
}

llvm::LogicalResult hlfir::RegionAssignOp::verify() {
  if (!aiir::isa_and_nonnull<hlfir::YieldOp>(getTerminator(getRhsRegion())))
    return emitOpError(
        "right-hand side region must be terminated by an hlfir.yield");
  if (!aiir::isa_and_nonnull<hlfir::YieldOp, hlfir::ElementalAddrOp>(
          getTerminator(getLhsRegion())))
    return emitOpError("left-hand side region must be terminated by an "
                       "hlfir.yield or hlfir.elemental_addr");
  return aiir::success();
}

static aiir::Type
getNonVectorSubscriptedLhsType(hlfir::RegionAssignOp regionAssign) {
  hlfir::YieldOp yieldOp = aiir::dyn_cast_or_null<hlfir::YieldOp>(
      getTerminator(regionAssign.getLhsRegion()));
  return yieldOp ? yieldOp.getEntity().getType() : aiir::Type{};
}

bool hlfir::RegionAssignOp::isPointerObjectAssignment() {
  if (!getUserDefinedAssignment().empty())
    return false;
  aiir::Type lhsType = getNonVectorSubscriptedLhsType(*this);
  return lhsType && hlfir::isFortranPointerObjectType(lhsType);
}

bool hlfir::RegionAssignOp::isProcedurePointerAssignment() {
  if (!getUserDefinedAssignment().empty())
    return false;
  aiir::Type lhsType = getNonVectorSubscriptedLhsType(*this);
  return lhsType && hlfir::isFortranProcedurePointerType(lhsType);
}

bool hlfir::RegionAssignOp::isPointerAssignment() {
  if (!getUserDefinedAssignment().empty())
    return false;
  aiir::Type lhsType = getNonVectorSubscriptedLhsType(*this);
  return lhsType && (hlfir::isFortranPointerObjectType(lhsType) ||
                     hlfir::isFortranProcedurePointerType(lhsType));
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static aiir::ParseResult parseYieldOpCleanup(aiir::OpAsmParser &parser,
                                             aiir::Region &cleanup) {
  if (succeeded(parser.parseOptionalKeyword("cleanup"))) {
    if (parser.parseRegion(cleanup, /*arguments=*/{},
                           /*argTypes=*/{}))
      return aiir::failure();
    hlfir::YieldOp::ensureTerminator(cleanup, parser.getBuilder(),
                                     parser.getBuilder().getUnknownLoc());
  }
  return aiir::success();
}

template <typename YieldOp>
static void printYieldOpCleanup(aiir::OpAsmPrinter &p, YieldOp yieldOp,
                                aiir::Region &cleanup) {
  if (!cleanup.empty()) {
    p << "cleanup ";
    p.printRegion(cleanup, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

//===----------------------------------------------------------------------===//
// ElementalAddrOp
//===----------------------------------------------------------------------===//

void hlfir::ElementalAddrOp::build(aiir::OpBuilder &builder,
                                   aiir::OperationState &odsState,
                                   aiir::Value shape, aiir::Value mold,
                                   aiir::ValueRange typeparams,
                                   bool isUnordered) {
  buildElemental<hlfir::ElementalAddrOp>(builder, odsState, shape, mold,
                                         typeparams, isUnordered);
  // Push cleanUp region.
  odsState.addRegion();
}

llvm::LogicalResult hlfir::ElementalAddrOp::verify() {
  hlfir::YieldOp yieldOp =
      aiir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(getBody()));
  if (!yieldOp)
    return emitOpError("body region must be terminated by an hlfir.yield");
  aiir::Type elementAddrType = yieldOp.getEntity().getType();
  if (!hlfir::isFortranVariableType(elementAddrType) ||
      aiir::isa<fir::SequenceType>(
          hlfir::getFortranElementOrSequenceType(elementAddrType)))
    return emitOpError("body must compute the address of a scalar entity");
  unsigned shapeRank =
      aiir::cast<fir::ShapeType>(getShape().getType()).getRank();
  if (shapeRank != getIndices().size())
    return emitOpError("body number of indices must match shape rank");
  return aiir::success();
}

hlfir::YieldOp hlfir::ElementalAddrOp::getYieldOp() {
  hlfir::YieldOp yieldOp =
      aiir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(getBody()));
  assert(yieldOp && "element_addr is ill-formed");
  return yieldOp;
}

aiir::Value hlfir::ElementalAddrOp::getElementEntity() {
  return getYieldOp().getEntity();
}

aiir::Region *hlfir::ElementalAddrOp::getElementCleanup() {
  aiir::Region *cleanup = &getYieldOp().getCleanup();
  return cleanup->empty() ? nullptr : cleanup;
}

//===----------------------------------------------------------------------===//
// OrderedAssignmentTreeOpInterface
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::OrderedAssignmentTreeOpInterface::verifyImpl() {
  if (aiir::Region *body = getSubTreeRegion())
    if (!body->empty())
      for (aiir::Operation &op : body->front())
        if (!aiir::isa<hlfir::OrderedAssignmentTreeOpInterface, fir::FirEndOp>(
                op))
          return emitOpError(
              "body region must only contain OrderedAssignmentTreeOpInterface "
              "operations or fir.end");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

static aiir::ParseResult parseForallOpBody(aiir::OpAsmParser &parser,
                                           aiir::Region &body) {
  aiir::OpAsmParser::Argument bodyArg;
  if (parser.parseLParen() || parser.parseArgument(bodyArg) ||
      parser.parseColon() || parser.parseType(bodyArg.type) ||
      parser.parseRParen())
    return aiir::failure();
  if (parser.parseRegion(body, {bodyArg}))
    return aiir::failure();
  ensureTerminator(body, parser.getBuilder(),
                   parser.getBuilder().getUnknownLoc());
  return aiir::success();
}

static void printForallOpBody(aiir::OpAsmPrinter &p, hlfir::ForallOp forall,
                              aiir::Region &body) {
  aiir::Value forallIndex = forall.getForallIndexValue();
  p << " (" << forallIndex << ": " << forallIndex.getType() << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

/// Predicate implementation of YieldIntegerOrEmpty.
static bool yieldsIntegerOrEmpty(aiir::Region &region) {
  if (region.empty())
    return true;
  auto yield = aiir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(region));
  return yield && fir::isa_integer(yield.getEntity().getType());
}

//===----------------------------------------------------------------------===//
// ForallMaskOp
//===----------------------------------------------------------------------===//

static aiir::ParseResult parseAssignmentMaskOpBody(aiir::OpAsmParser &parser,
                                                   aiir::Region &body) {
  if (parser.parseRegion(body))
    return aiir::failure();
  ensureTerminator(body, parser.getBuilder(),
                   parser.getBuilder().getUnknownLoc());
  return aiir::success();
}

template <typename ConcreteOp>
static void printAssignmentMaskOpBody(aiir::OpAsmPrinter &p, ConcreteOp,
                                      aiir::Region &body) {
  // ElseWhereOp is a WhereOp/ElseWhereOp terminator that should be printed.
  bool printBlockTerminators =
      !body.empty() &&
      aiir::isa_and_nonnull<hlfir::ElseWhereOp>(body.back().getTerminator());
  p.printRegion(body, /*printEntryBlockArgs=*/false, printBlockTerminators);
}

static bool yieldsLogical(aiir::Region &region, bool mustBeScalarI1) {
  if (region.empty())
    return false;
  auto yield = aiir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(region));
  if (!yield)
    return false;
  aiir::Type yieldType = yield.getEntity().getType();
  if (mustBeScalarI1)
    return hlfir::isI1Type(yieldType);
  return hlfir::isMaskArgument(yieldType) &&
         aiir::isa<fir::SequenceType>(
             hlfir::getFortranElementOrSequenceType(yieldType));
}

llvm::LogicalResult hlfir::ForallMaskOp::verify() {
  if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/true))
    return emitOpError("mask region must yield a scalar i1");
  aiir::Operation *op = getOperation();
  hlfir::ForallOp forallOp =
      aiir::dyn_cast_or_null<hlfir::ForallOp>(op->getParentOp());
  if (!forallOp || op->getParentRegion() != &forallOp.getBody())
    return emitOpError("must be inside the body region of an hlfir.forall");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// WhereOp and ElseWhereOp
//===----------------------------------------------------------------------===//

template <typename ConcreteOp>
static llvm::LogicalResult verifyWhereAndElseWhereBody(ConcreteOp &concreteOp) {
  for (aiir::Operation &op : concreteOp.getBody().front())
    if (aiir::isa<hlfir::ForallOp>(op))
      return concreteOp.emitOpError(
          "body region must not contain hlfir.forall");
  return aiir::success();
}

llvm::LogicalResult hlfir::WhereOp::verify() {
  if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/false))
    return emitOpError("mask region must yield a logical array");
  return verifyWhereAndElseWhereBody(*this);
}

llvm::LogicalResult hlfir::ElseWhereOp::verify() {
  if (!getMaskRegion().empty())
    if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/false))
      return emitOpError(
          "mask region must yield a logical array when provided");
  return verifyWhereAndElseWhereBody(*this);
}

//===----------------------------------------------------------------------===//
// ForallIndexOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult
hlfir::ForallIndexOp::canonicalize(hlfir::ForallIndexOp indexOp,
                                   aiir::PatternRewriter &rewriter) {
  for (aiir::Operation *user : indexOp->getResult(0).getUsers())
    if (!aiir::isa<fir::LoadOp>(user))
      return aiir::failure();

  auto insertPt = rewriter.saveInsertionPoint();
  llvm::SmallVector<aiir::Operation *> users(indexOp->getResult(0).getUsers());
  for (aiir::Operation *user : users)
    if (auto loadOp = aiir::dyn_cast<fir::LoadOp>(user)) {
      rewriter.setInsertionPoint(loadOp);
      rewriter.replaceOpWithNewOp<fir::ConvertOp>(
          user, loadOp.getResult().getType(), indexOp.getIndex());
    }
  rewriter.restoreInsertionPoint(insertPt);
  rewriter.eraseOp(indexOp);
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CharExtremumOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult hlfir::CharExtremumOp::verify() {
  if (getStrings().size() < 2)
    return emitOpError("must be provided at least two string operands");
  unsigned kind = getCharacterKind(getResult().getType());
  for (auto string : getStrings())
    if (kind != getCharacterKind(string.getType()))
      return emitOpError("strings must have the same KIND as the result type");
  return aiir::success();
}

void hlfir::CharExtremumOp::build(aiir::OpBuilder &builder,
                                  aiir::OperationState &result,
                                  hlfir::CharExtremumPredicate predicate,
                                  aiir::ValueRange strings) {

  fir::CharacterType::LenType resultTypeLen = 0;
  assert(!strings.empty() && "must contain operands");
  unsigned kind = getCharacterKind(strings[0].getType());
  for (auto string : strings)
    if (auto cstLen = getCharacterLengthIfStatic(string.getType())) {
      resultTypeLen = std::max(resultTypeLen, *cstLen);
    } else {
      resultTypeLen = fir::CharacterType::unknownLen();
      break;
    }
  auto resultType = hlfir::ExprType::get(
      builder.getContext(), hlfir::ExprType::Shape{},
      fir::CharacterType::get(builder.getContext(), kind, resultTypeLen),
      false);

  build(builder, result, resultType, predicate, strings);
}

void hlfir::CharExtremumOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// GetLength
//===----------------------------------------------------------------------===//

llvm::LogicalResult
hlfir::GetLengthOp::canonicalize(GetLengthOp getLength,
                                 aiir::PatternRewriter &rewriter) {
  aiir::Location loc = getLength.getLoc();
  auto exprTy = aiir::cast<hlfir::ExprType>(getLength.getExpr().getType());
  auto charTy = aiir::cast<fir::CharacterType>(exprTy.getElementType());
  if (!charTy.hasConstantLen())
    return aiir::failure();

  aiir::Type indexTy = rewriter.getIndexType();
  auto cstLen = aiir::arith::ConstantOp::create(
      rewriter, loc, indexTy, aiir::IntegerAttr::get(indexTy, charTy.getLen()));
  rewriter.replaceOp(getLength, cstLen);
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// EvaluateInMemoryOp
//===----------------------------------------------------------------------===//

void hlfir::EvaluateInMemoryOp::build(aiir::OpBuilder &builder,
                                      aiir::OperationState &odsState,
                                      aiir::Type resultType, aiir::Value shape,
                                      aiir::ValueRange typeparams) {
  odsState.addTypes(resultType);
  if (shape)
    odsState.addOperands(shape);
  odsState.addOperands(typeparams);
  odsState.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {shape ? 1 : 0, static_cast<int32_t>(typeparams.size())}));
  aiir::Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new aiir::Block{});
  aiir::Type memType = fir::ReferenceType::get(
      hlfir::getFortranElementOrSequenceType(resultType));
  bodyRegion->front().addArgument(memType, odsState.location);
  EvaluateInMemoryOp::ensureTerminator(*bodyRegion, builder, odsState.location);
}

llvm::LogicalResult hlfir::EvaluateInMemoryOp::verify() {
  unsigned shapeRank = 0;
  if (aiir::Value shape = getShape())
    if (auto shapeTy = aiir::dyn_cast<fir::ShapeType>(shape.getType()))
      shapeRank = shapeTy.getRank();
  auto exprType = aiir::cast<hlfir::ExprType>(getResult().getType());
  if (shapeRank != exprType.getRank())
    return emitOpError("`shape` rank must match the result rank");
  aiir::Type elementType = exprType.getElementType();
  if (auto res = verifyTypeparams(*this, elementType, getTypeparams().size());
      failed(res))
    return res;
  return aiir::success();
}

#include "flang/Optimizer/HLFIR/HLFIROpInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIREnums.cpp.inc"
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
