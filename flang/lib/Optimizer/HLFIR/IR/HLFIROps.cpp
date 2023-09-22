//===-- HLFIROps.cpp ------------------------------------------------------===//
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

#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <tuple>

/// generic implementation of the memory side effects interface for hlfir
/// transformational intrinsic operations
static void
getIntrinsicEffects(mlir::Operation *self,
                    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                        mlir::MemoryEffects::Effect>> &effects) {
  // allocation effect if we return an expr
  assert(self->getNumResults() == 1 &&
         "hlfir intrinsic ops only produce 1 result");
  if (mlir::isa<hlfir::ExprType>(self->getResult(0).getType()))
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(),
                         self->getResult(0),
                         mlir::SideEffects::DefaultResource::get());

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
  for (mlir::Value operand : self->getOperands()) {
    mlir::Type opTy = operand.getType();
    if (fir::isa_ref_type(opTy) || fir::isa_box_type(opTy))
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand,
                           mlir::SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

/// Is this a fir.[ref/ptr/heap]<fir.[box/class]<fir.heap<T>>> type?
static bool isAllocatableBoxRef(mlir::Type type) {
  fir::BaseBoxType boxType =
      fir::dyn_cast_ptrEleTy(type).dyn_cast_or_null<fir::BaseBoxType>();
  return boxType && boxType.getEleTy().isa<fir::HeapType>();
}

mlir::LogicalResult hlfir::AssignOp::verify() {
  mlir::Type lhsType = getLhs().getType();
  if (isAllocatableAssignment() && !isAllocatableBoxRef(lhsType))
    return emitOpError("lhs must be an allocatable when `realloc` is set");
  if (mustKeepLhsLengthInAllocatableAssignment() &&
      !(isAllocatableAssignment() &&
        hlfir::getFortranElementType(lhsType).isa<fir::CharacterType>()))
    return emitOpError("`realloc` must be set and lhs must be a character "
                       "allocatable when `keep_lhs_length_if_realloc` is set");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

/// Given a FIR memory type, and information about non default lower bounds, get
/// the related HLFIR variable type.
mlir::Type hlfir::DeclareOp::getHLFIRVariableType(mlir::Type inputType,
                                                  bool hasExplicitLowerBounds) {
  mlir::Type type = fir::unwrapRefType(inputType);
  if (type.isa<fir::BaseBoxType>())
    return inputType;
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());

  auto seqType = type.dyn_cast<fir::SequenceType>();
  bool hasDynamicExtents =
      seqType && fir::sequenceWithNonConstantShape(seqType);
  mlir::Type eleType = seqType ? seqType.getEleTy() : type;
  bool hasDynamicLengthParams = fir::characterWithDynamicLen(eleType) ||
                                fir::isRecordWithTypeParameters(eleType);
  if (hasExplicitLowerBounds || hasDynamicExtents || hasDynamicLengthParams)
    return fir::BoxType::get(type);
  return inputType;
}

static bool hasExplicitLowerBounds(mlir::Value shape) {
  return shape && shape.getType().isa<fir::ShapeShiftType, fir::ShiftType>();
}

void hlfir::DeclareOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value memref,
                             llvm::StringRef uniq_name, mlir::Value shape,
                             mlir::ValueRange typeparams,
                             fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  mlir::Type inputType = memref.getType();
  bool hasExplicitLbs = hasExplicitLowerBounds(shape);
  mlir::Type hlfirVariableType =
      getHLFIRVariableType(inputType, hasExplicitLbs);
  build(builder, result, {hlfirVariableType, inputType}, memref, shape,
        typeparams, nameAttr, fortran_attrs);
}

mlir::LogicalResult hlfir::DeclareOp::verify() {
  if (getMemref().getType() != getResult(1).getType())
    return emitOpError("second result type must match input memref type");
  mlir::Type hlfirVariableType = getHLFIRVariableType(
      getMemref().getType(), hasExplicitLowerBounds(getShape()));
  if (hlfirVariableType != getResult(0).getType())
    return emitOpError("first result type is inconsistent with variable "
                       "properties: expected ")
           << hlfirVariableType;
  // The rest of the argument verification is done by the
  // FortranVariableInterface verifier.
  auto fortranVar =
      mlir::cast<fir::FortranVariableOpInterface>(this->getOperation());
  return fortranVar.verifyDeclareLikeOpImpl(getMemref());
}

//===----------------------------------------------------------------------===//
// DesignateOp
//===----------------------------------------------------------------------===//

void hlfir::DesignateOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    mlir::Type result_type, mlir::Value memref, llvm::StringRef component,
    mlir::Value component_shape, llvm::ArrayRef<Subscript> subscripts,
    mlir::ValueRange substring, std::optional<bool> complex_part,
    mlir::Value shape, mlir::ValueRange typeparams,
    fir::FortranVariableFlagsAttr fortran_attrs) {
  auto componentAttr =
      component.empty() ? mlir::StringAttr{} : builder.getStringAttr(component);
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<bool> isTriplet;
  for (auto subscript : subscripts) {
    if (auto *triplet = std::get_if<Triplet>(&subscript)) {
      isTriplet.push_back(true);
      indices.push_back(std::get<0>(*triplet));
      indices.push_back(std::get<1>(*triplet));
      indices.push_back(std::get<2>(*triplet));
    } else {
      isTriplet.push_back(false);
      indices.push_back(std::get<mlir::Value>(subscript));
    }
  }
  auto isTripletAttr =
      mlir::DenseBoolArrayAttr::get(builder.getContext(), isTriplet);
  auto complexPartAttr =
      complex_part.has_value()
          ? mlir::BoolAttr::get(builder.getContext(), *complex_part)
          : mlir::BoolAttr{};
  build(builder, result, result_type, memref, componentAttr, component_shape,
        indices, isTripletAttr, substring, complexPartAttr, shape, typeparams,
        fortran_attrs);
}

void hlfir::DesignateOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &result,
                               mlir::Type result_type, mlir::Value memref,
                               mlir::ValueRange indices,
                               mlir::ValueRange typeparams,
                               fir::FortranVariableFlagsAttr fortran_attrs) {
  llvm::SmallVector<bool> isTriplet(indices.size(), false);
  auto isTripletAttr =
      mlir::DenseBoolArrayAttr::get(builder.getContext(), isTriplet);
  build(builder, result, result_type, memref,
        /*componentAttr=*/mlir::StringAttr{}, /*component_shape=*/mlir::Value{},
        indices, isTripletAttr, /*substring*/ mlir::ValueRange{},
        /*complexPartAttr=*/mlir::BoolAttr{}, /*shape=*/mlir::Value{},
        typeparams, fortran_attrs);
}

static mlir::ParseResult parseDesignatorIndices(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &indices,
    mlir::DenseBoolArrayAttr &isTripletAttr) {
  llvm::SmallVector<bool> isTriplet;
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    do {
      mlir::OpAsmParser::UnresolvedOperand i1, i2, i3;
      if (parser.parseOperand(i1))
        return mlir::failure();
      indices.push_back(i1);
      if (mlir::succeeded(parser.parseOptionalColon())) {
        if (parser.parseOperand(i2) || parser.parseColon() ||
            parser.parseOperand(i3))
          return mlir::failure();
        indices.push_back(i2);
        indices.push_back(i3);
        isTriplet.push_back(true);
      } else {
        isTriplet.push_back(false);
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return mlir::failure();
  }
  isTripletAttr = mlir::DenseBoolArrayAttr::get(parser.getContext(), isTriplet);
  return mlir::success();
}

static void
printDesignatorIndices(mlir::OpAsmPrinter &p, hlfir::DesignateOp designateOp,
                       mlir::OperandRange indices,
                       const mlir::DenseBoolArrayAttr &isTripletAttr) {
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

static mlir::ParseResult
parseDesignatorComplexPart(mlir::OpAsmParser &parser,
                           mlir::BoolAttr &complexPart) {
  if (mlir::succeeded(parser.parseOptionalKeyword("imag")))
    complexPart = mlir::BoolAttr::get(parser.getContext(), true);
  else if (mlir::succeeded(parser.parseOptionalKeyword("real")))
    complexPart = mlir::BoolAttr::get(parser.getContext(), false);
  return mlir::success();
}

static void printDesignatorComplexPart(mlir::OpAsmPrinter &p,
                                       hlfir::DesignateOp designateOp,
                                       mlir::BoolAttr complexPartAttr) {
  if (complexPartAttr) {
    if (complexPartAttr.getValue())
      p << "imag";
    else
      p << "real";
  }
}

mlir::LogicalResult hlfir::DesignateOp::verify() {
  mlir::Type memrefType = getMemref().getType();
  mlir::Type baseType = getFortranElementOrSequenceType(memrefType);
  mlir::Type baseElementType = fir::unwrapSequenceType(baseType);
  unsigned numSubscripts = getIsTriplet().size();
  unsigned subscriptsRank =
      llvm::count_if(getIsTriplet(), [](bool isTriplet) { return isTriplet; });
  unsigned outputRank = 0;
  mlir::Type outputElementType;
  bool hasBoxComponent;
  if (getComponent()) {
    auto component = getComponent().value();
    auto recType = baseElementType.dyn_cast<fir::RecordType>();
    if (!recType)
      return emitOpError(
          "component must be provided only when the memref is a derived type");
    unsigned fieldIdx = recType.getFieldIndex(component);
    if (fieldIdx > recType.getNumFields()) {
      return emitOpError("component ")
             << component << " is not a component of memref element type "
             << recType;
    }
    mlir::Type fieldType = recType.getType(fieldIdx);
    mlir::Type componentBaseType = getFortranElementOrSequenceType(fieldType);
    hasBoxComponent = fieldType.isa<fir::BaseBoxType>();
    if (componentBaseType.isa<fir::SequenceType>() &&
        baseType.isa<fir::SequenceType>() &&
        (numSubscripts == 0 || subscriptsRank > 0))
      return emitOpError("indices must be provided and must not contain "
                         "triplets when both memref and component are arrays");
    if (numSubscripts != 0) {
      if (!componentBaseType.isa<fir::SequenceType>())
        return emitOpError("indices must not be provided if component appears "
                           "and is not an array component");
      if (!getComponentShape())
        return emitOpError(
            "component_shape must be provided when indexing a component");
      mlir::Type compShapeType = getComponentShape().getType();
      unsigned componentRank =
          componentBaseType.cast<fir::SequenceType>().getDimension();
      auto shapeType = compShapeType.dyn_cast<fir::ShapeType>();
      auto shapeShiftType = compShapeType.dyn_cast<fir::ShapeShiftType>();
      if (!((shapeType && shapeType.getRank() == componentRank) ||
            (shapeShiftType && shapeShiftType.getRank() == componentRank)))
        return emitOpError("component_shape must be a fir.shape or "
                           "fir.shapeshift with the rank of the component");
      if (numSubscripts > componentRank)
        return emitOpError("indices number must match array component rank");
    }
    if (auto baseSeqType = baseType.dyn_cast<fir::SequenceType>())
      // This case must come first to cover "array%array_comp(i, j)" that has
      // subscripts for the component but whose rank come from the base.
      outputRank = baseSeqType.getDimension();
    else if (numSubscripts != 0)
      outputRank = subscriptsRank;
    else if (auto componentSeqType =
                 componentBaseType.dyn_cast<fir::SequenceType>())
      outputRank = componentSeqType.getDimension();
    outputElementType = fir::unwrapSequenceType(componentBaseType);
  } else {
    outputElementType = baseElementType;
    unsigned baseTypeRank =
        baseType.isa<fir::SequenceType>()
            ? baseType.cast<fir::SequenceType>().getDimension()
            : 0;
    if (numSubscripts != 0) {
      if (baseTypeRank != numSubscripts)
        return emitOpError("indices number must match memref rank");
      outputRank = subscriptsRank;
    } else if (auto baseSeqType = baseType.dyn_cast<fir::SequenceType>()) {
      outputRank = baseSeqType.getDimension();
    }
  }

  if (!getSubstring().empty()) {
    if (!outputElementType.isa<fir::CharacterType>())
      return emitOpError("memref or component must have character type if "
                         "substring indices are provided");
    if (getSubstring().size() != 2)
      return emitOpError("substring must contain 2 indices when provided");
  }
  if (getComplexPart()) {
    if (!fir::isa_complex(outputElementType))
      return emitOpError("memref or component must have complex type if "
                         "complex_part is provided");
    if (auto firCplx = outputElementType.dyn_cast<fir::ComplexType>())
      outputElementType = firCplx.getElementType();
    else
      outputElementType =
          outputElementType.cast<mlir::ComplexType>().getElementType();
  }
  mlir::Type resultBaseType =
      getFortranElementOrSequenceType(getResult().getType());
  unsigned resultRank = 0;
  if (auto resultSeqType = resultBaseType.dyn_cast<fir::SequenceType>())
    resultRank = resultSeqType.getDimension();
  if (resultRank != outputRank)
    return emitOpError("result type rank is not consistent with operands, "
                       "expected rank ")
           << outputRank;
  mlir::Type resultElementType = fir::unwrapSequenceType(resultBaseType);
  // result type must match the one that was inferred here, except the character
  // length may differ because of substrings.
  if (resultElementType != outputElementType &&
      !(resultElementType.isa<fir::CharacterType>() &&
        outputElementType.isa<fir::CharacterType>()) &&
      !(resultElementType.isa<mlir::FloatType>() &&
        outputElementType.isa<fir::RealType>()))
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
      auto shapeType = getShape().getType().dyn_cast<fir::ShapeType>();
      auto shapeShiftType =
          getShape().getType().dyn_cast<fir::ShapeShiftType>();
      if (!((shapeType && shapeType.getRank() == resultRank) ||
            (shapeShiftType && shapeShiftType.getRank() == resultRank)))
        return emitOpError("shape must be a fir.shape or fir.shapeshift with "
                           "the rank of the result");
    }
    auto numLenParam = getTypeparams().size();
    if (outputElementType.isa<fir::CharacterType>()) {
      if (numLenParam != 1)
        return emitOpError("must be provided one length parameter when the "
                           "result is a character");
    } else if (fir::isRecordWithTypeParameters(outputElementType)) {
      if (numLenParam !=
          outputElementType.cast<fir::RecordType>().getNumLenParams())
        return emitOpError("must be provided the same number of length "
                           "parameters as in the result derived type");
    } else if (numLenParam != 0) {
      return emitOpError("must not be provided length parameters if the result "
                         "type does not have length parameters");
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ParentComponentOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::ParentComponentOp::verify() {
  mlir::Type baseType =
      hlfir::getFortranElementOrSequenceType(getMemref().getType());
  auto maybeInputSeqType = baseType.dyn_cast<fir::SequenceType>();
  unsigned inputTypeRank =
      maybeInputSeqType ? maybeInputSeqType.getDimension() : 0;
  unsigned shapeRank = 0;
  if (mlir::Value shape = getShape())
    if (auto shapeType = shape.getType().dyn_cast<fir::ShapeType>())
      shapeRank = shapeType.getRank();
  if (inputTypeRank != shapeRank)
    return emitOpError(
        "must be provided a shape if and only if the base is an array");
  mlir::Type outputBaseType = hlfir::getFortranElementOrSequenceType(getType());
  auto maybeOutputSeqType = outputBaseType.dyn_cast<fir::SequenceType>();
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
      hlfir::getFortranElementType(baseType).dyn_cast<fir::RecordType>();
  fir::RecordType outRecType =
      hlfir::getFortranElementType(outputBaseType).dyn_cast<fir::RecordType>();
  if (!baseRecType || !outRecType)
    return emitOpError("result type and input type must be derived types");

  // Note: result should not be a fir.class: its dynamic type is being set to
  // the parent type and allowing fir.class would break the operation codegen:
  // it would keep the input dynamic type.
  if (getType().isa<fir::ClassType>())
    return emitOpError("result type must not be polymorphic");

  // The array results are known to not be dis-contiguous in most cases (the
  // exception being if the parent type was extended by a type without any
  // components): require a fir.box to be used for the result to carry the
  // strides.
  if (!getType().isa<fir::BoxType>() &&
      (outputTypeRank != 0 || fir::isRecordWithTypeParameters(outRecType)))
    return emitOpError("result type must be a fir.box if the result is an "
                       "array or has length parameters");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LogicalReductionOp
//===----------------------------------------------------------------------===//
template <typename LogicalReductionOp>
static mlir::LogicalResult
verifyLogicalReductionOp(LogicalReductionOp reductionOp) {
  mlir::Operation *op = reductionOp->getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  mlir::Value mask = reductionOp->getMask();
  mlir::Value dim = reductionOp->getDim();

  fir::SequenceType maskTy =
      hlfir::getFortranElementOrSequenceType(mask.getType())
          .cast<fir::SequenceType>();
  mlir::Type logicalTy = maskTy.getEleTy();
  llvm::ArrayRef<int64_t> maskShape = maskTy.getShape();

  mlir::Type resultType = results[0];
  if (mlir::isa<fir::LogicalType>(resultType)) {
    // Result is of the same type as MASK
    if (resultType != logicalTy)
      return reductionOp->emitOpError(
          "result must have the same element type as MASK argument");

  } else if (auto resultExpr =
                 mlir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    // Result should only be in hlfir.expr form if it is an array
    if (maskShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return reductionOp->emitOpError("result must be an array");

      if (resultExpr.getEleTy() != logicalTy)
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
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AllOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::AllOp::verify() {
  return verifyLogicalReductionOp<hlfir::AllOp *>(this);
}

void hlfir::AllOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// AnyOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::AnyOp::verify() {
  return verifyLogicalReductionOp<hlfir::AnyOp *>(this);
}

void hlfir::AnyOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CountOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::CountOp::verify() {
  mlir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);
  mlir::Value mask = getMask();
  mlir::Value dim = getDim();

  fir::SequenceType maskTy =
      hlfir::getFortranElementOrSequenceType(mask.getType())
          .cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> maskShape = maskTy.getShape();

  mlir::Type resultType = results[0];
  if (auto resultExpr = mlir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    if (maskShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return emitOpError("result must be an array");

      llvm::ArrayRef<int64_t> resultShape = resultExpr.getShape();
      // Result has rank n-1
      if (resultShape.size() != (maskShape.size() - 1))
        return emitOpError("result rank must be one less than MASK");
    } else {
      return emitOpError("result must be of numerical scalar type");
    }
  } else if (!hlfir::isFortranScalarNumericalType(resultType)) {
    return emitOpError("result must be of numerical scalar type");
  }

  return mlir::success();
}

void hlfir::CountOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getCharacterKind(mlir::Type t) {
  return hlfir::getFortranElementType(t).cast<fir::CharacterType>().getFKind();
}

static std::optional<fir::CharacterType::LenType>
getCharacterLengthIfStatic(mlir::Type t) {
  if (auto charType =
          hlfir::getFortranElementType(t).dyn_cast<fir::CharacterType>())
    if (charType.hasConstantLen())
      return charType.getLen();
  return std::nullopt;
}

mlir::LogicalResult hlfir::ConcatOp::verify() {
  if (getStrings().size() < 2)
    return emitOpError("must be provided at least two string operands");
  unsigned kind = getCharacterKind(getResult().getType());
  for (auto string : getStrings())
    if (kind != getCharacterKind(string.getType()))
      return emitOpError("strings must have the same KIND as the result type");
  return mlir::success();
}

void hlfir::ConcatOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result,
                            mlir::ValueRange strings, mlir::Value len) {
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
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// NumericalReductionOp
//===----------------------------------------------------------------------===//

template <typename NumericalReductionOp>
static mlir::LogicalResult
verifyNumericalReductionOp(NumericalReductionOp reductionOp) {
  mlir::Operation *op = reductionOp->getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  mlir::Value array = reductionOp->getArray();
  mlir::Value dim = reductionOp->getDim();
  mlir::Value mask = reductionOp->getMask();

  fir::SequenceType arrayTy =
      hlfir::getFortranElementOrSequenceType(array.getType())
          .cast<fir::SequenceType>();
  mlir::Type numTy = arrayTy.getEleTy();
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  if (mask) {
    fir::SequenceType maskSeq =
        hlfir::getFortranElementOrSequenceType(mask.getType())
            .dyn_cast<fir::SequenceType>();
    llvm::ArrayRef<int64_t> maskShape;

    if (maskSeq)
      maskShape = maskSeq.getShape();

    if (!maskShape.empty()) {
      if (maskShape.size() != arrayShape.size())
        return reductionOp->emitWarning("MASK must be conformable to ARRAY");
      static_assert(fir::SequenceType::getUnknownExtent() ==
                    hlfir::ExprType::getUnknownExtent());
      constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
      for (std::size_t i = 0; i < arrayShape.size(); ++i) {
        int64_t arrayExtent = arrayShape[i];
        int64_t maskExtent = maskShape[i];
        if ((arrayExtent != maskExtent) && (arrayExtent != unknownExtent) &&
            (maskExtent != unknownExtent))
          return reductionOp->emitWarning("MASK must be conformable to ARRAY");
      }
    }
  }

  mlir::Type resultType = results[0];
  if (hlfir::isFortranScalarNumericalType(resultType)) {
    // Result is of the same type as ARRAY
    if (resultType != numTy)
      return reductionOp->emitOpError(
          "result must have the same element type as ARRAY argument");

  } else if (auto resultExpr =
                 mlir::dyn_cast_or_null<hlfir::ExprType>(resultType)) {
    if (arrayShape.size() > 1 && dim != nullptr) {
      if (!resultExpr.isArray())
        return reductionOp->emitOpError("result must be an array");

      if (resultExpr.getEleTy() != numTy)
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
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::ProductOp::verify() {
  return verifyNumericalReductionOp<hlfir::ProductOp *>(this);
}

void hlfir::ProductOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// CharacterReductionOp
//===----------------------------------------------------------------------===//

template <typename CharacterReductionOp>
static mlir::LogicalResult
verifyCharacterReductionOp(CharacterReductionOp reductionOp) {
  mlir::Operation *op = reductionOp->getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  mlir::Value array = reductionOp->getArray();
  mlir::Value dim = reductionOp->getDim();
  mlir::Value mask = reductionOp->getMask();

  fir::SequenceType arrayTy =
      hlfir::getFortranElementOrSequenceType(array.getType())
          .cast<fir::SequenceType>();
  mlir::Type numTy = arrayTy.getEleTy();
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();

  if (mask) {
    fir::SequenceType maskSeq =
        hlfir::getFortranElementOrSequenceType(mask.getType())
            .dyn_cast<fir::SequenceType>();
    llvm::ArrayRef<int64_t> maskShape;

    if (maskSeq)
      maskShape = maskSeq.getShape();

    if (!maskShape.empty()) {
      if (maskShape.size() != arrayShape.size())
        return reductionOp->emitWarning("MASK must be conformable to ARRAY");
      static_assert(fir::SequenceType::getUnknownExtent() ==
                    hlfir::ExprType::getUnknownExtent());
      constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
      for (std::size_t i = 0; i < arrayShape.size(); ++i) {
        int64_t arrayExtent = arrayShape[i];
        int64_t maskExtent = maskShape[i];
        if ((arrayExtent != maskExtent) && (arrayExtent != unknownExtent) &&
            (maskExtent != unknownExtent))
          return reductionOp->emitWarning("MASK must be conformable to ARRAY");
      }
    }
  }

  auto resultExpr = results[0].cast<hlfir::ExprType>();
  mlir::Type resultType = resultExpr.getEleTy();
  assert(mlir::isa<fir::CharacterType>(resultType) &&
         "result must be character");

  // Result is of the same type as ARRAY
  if (resultType != numTy)
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
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaxvalOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::MaxvalOp::verify() {
  mlir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto resultExpr = mlir::dyn_cast<hlfir::ExprType>(results[0]);
  if (resultExpr && mlir::isa<fir::CharacterType>(resultExpr.getEleTy())) {
    return verifyCharacterReductionOp<hlfir::MaxvalOp *>(this);
  } else {
    return verifyNumericalReductionOp<hlfir::MaxvalOp *>(this);
  }
}

void hlfir::MaxvalOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MinvalOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::MinvalOp::verify() {
  mlir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  auto resultExpr = mlir::dyn_cast<hlfir::ExprType>(results[0]);
  if (resultExpr && mlir::isa<fir::CharacterType>(resultExpr.getEleTy())) {
    return verifyCharacterReductionOp<hlfir::MinvalOp *>(this);
  } else {
    return verifyNumericalReductionOp<hlfir::MinvalOp *>(this);
  }
}

void hlfir::MinvalOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// SetLengthOp
//===----------------------------------------------------------------------===//

void hlfir::SetLengthOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &result, mlir::Value string,
                               mlir::Value len) {
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
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::SumOp::verify() {
  return verifyNumericalReductionOp<hlfir::SumOp *>(this);
}

void hlfir::SumOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// DotProductOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::DotProductOp::verify() {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();
  fir::SequenceType lhsTy =
      hlfir::getFortranElementOrSequenceType(lhs.getType())
          .cast<fir::SequenceType>();
  fir::SequenceType rhsTy =
      hlfir::getFortranElementOrSequenceType(rhs.getType())
          .cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  mlir::Type lhsEleTy = lhsTy.getEleTy();
  mlir::Type rhsEleTy = rhsTy.getEleTy();
  mlir::Type resultTy = getResult().getType();

  if ((lhsRank != 1) || (rhsRank != 1))
    return emitOpError("both arrays must have rank 1");

  int64_t lhsSize = lhsShape[0];
  int64_t rhsSize = rhsShape[0];

  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if ((lhsSize != unknownExtent) && (rhsSize != unknownExtent) &&
      (lhsSize != rhsSize))
    return emitOpError("both arrays must have the same size");

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(rhsEleTy))
    return emitOpError("if one array is logical, so should the other be");

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(resultTy))
    return emitOpError("the result type should be a logical only if the "
                       "argument types are logical");

  if (!hlfir::isFortranScalarNumericalType(resultTy) &&
      !mlir::isa<fir::LogicalType>(resultTy))
    return emitOpError(
        "the result must be of scalar numerical or logical type");

  return mlir::success();
}

void hlfir::DotProductOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::MatmulOp::verify() {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();
  fir::SequenceType lhsTy =
      hlfir::getFortranElementOrSequenceType(lhs.getType())
          .cast<fir::SequenceType>();
  fir::SequenceType rhsTy =
      hlfir::getFortranElementOrSequenceType(rhs.getType())
          .cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  mlir::Type lhsEleTy = lhsTy.getEleTy();
  mlir::Type rhsEleTy = rhsTy.getEleTy();
  hlfir::ExprType resultTy = getResult().getType().cast<hlfir::ExprType>();
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  mlir::Type resultEleTy = resultTy.getEleTy();

  if (((lhsRank != 1) && (lhsRank != 2)) || ((rhsRank != 1) && (rhsRank != 2)))
    return emitOpError("array must have either rank 1 or rank 2");

  if ((lhsRank == 1) && (rhsRank == 1))
    return emitOpError("at least one array must have rank 2");

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(rhsEleTy))
    return emitOpError("if one array is logical, so should the other be");

  int64_t lastLhsDim = lhsShape[lhsRank - 1];
  int64_t firstRhsDim = rhsShape[0];
  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if (lastLhsDim != firstRhsDim)
    if ((lastLhsDim != unknownExtent) && (firstRhsDim != unknownExtent))
      return emitOpError(
          "the last dimension of LHS should match the first dimension of RHS");

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(resultEleTy))
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

  return mlir::success();
}

mlir::LogicalResult
hlfir::MatmulOp::canonicalize(MatmulOp matmulOp,
                              mlir::PatternRewriter &rewriter) {
  // the only two uses of the transposed matrix should be for the hlfir.matmul
  // and hlfir.destory
  auto isOtherwiseUnused = [&](hlfir::TransposeOp transposeOp) -> bool {
    std::size_t numUses = 0;
    for (mlir::Operation *user : transposeOp.getResult().getUsers()) {
      ++numUses;
      if (user == matmulOp)
        continue;
      if (mlir::dyn_cast_or_null<hlfir::DestroyOp>(user))
        continue;
      // some other use!
      return false;
    }
    return numUses <= 2;
  };

  mlir::Value lhs = matmulOp.getLhs();
  // Rewrite MATMUL(TRANSPOSE(lhs), rhs) => hlfir.matmul_transpose lhs, rhs
  if (auto transposeOp = lhs.getDefiningOp<hlfir::TransposeOp>()) {
    if (isOtherwiseUnused(transposeOp)) {
      mlir::Location loc = matmulOp.getLoc();
      mlir::Type resultTy = matmulOp.getResult().getType();
      auto matmulTransposeOp = rewriter.create<hlfir::MatmulTransposeOp>(
          loc, resultTy, transposeOp.getArray(), matmulOp.getRhs());

      // we don't need to remove any hlfir.destroy because it will be needed for
      // the new intrinsic result anyway
      rewriter.replaceOp(matmulOp, matmulTransposeOp.getResult());

      // but we do need to get rid of the hlfir.destroy for the hlfir.transpose
      // result (which is entirely removed)
      for (mlir::Operation *user : transposeOp->getResult(0).getUsers())
        if (auto destroyOp = mlir::dyn_cast_or_null<hlfir::DestroyOp>(user))
          rewriter.eraseOp(destroyOp);
      rewriter.eraseOp(transposeOp);

      return mlir::success();
    }
  }

  return mlir::failure();
}

void hlfir::MatmulOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::TransposeOp::verify() {
  mlir::Value array = getArray();
  fir::SequenceType arrayTy =
      hlfir::getFortranElementOrSequenceType(array.getType())
          .cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> inShape = arrayTy.getShape();
  std::size_t rank = inShape.size();
  mlir::Type eleTy = arrayTy.getEleTy();
  hlfir::ExprType resultTy = getResult().getType().cast<hlfir::ExprType>();
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  std::size_t resultRank = resultShape.size();
  mlir::Type resultEleTy = resultTy.getEleTy();

  if (rank != 2 || resultRank != 2)
    return emitOpError("input and output arrays should have rank 2");

  constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
  if ((inShape[0] != resultShape[1]) && (inShape[0] != unknownExtent))
    return emitOpError("output shape does not match input array");
  if ((inShape[1] != resultShape[0]) && (inShape[1] != unknownExtent))
    return emitOpError("output shape does not match input array");

  if (eleTy != resultEleTy)
    return emitOpError(
        "input and output arrays should have the same element type");

  return mlir::success();
}

void hlfir::TransposeOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// MatmulTransposeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::MatmulTransposeOp::verify() {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();
  fir::SequenceType lhsTy =
      hlfir::getFortranElementOrSequenceType(lhs.getType())
          .cast<fir::SequenceType>();
  fir::SequenceType rhsTy =
      hlfir::getFortranElementOrSequenceType(rhs.getType())
          .cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  std::size_t lhsRank = lhsShape.size();
  std::size_t rhsRank = rhsShape.size();
  mlir::Type lhsEleTy = lhsTy.getEleTy();
  mlir::Type rhsEleTy = rhsTy.getEleTy();
  hlfir::ExprType resultTy = getResult().getType().cast<hlfir::ExprType>();
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  mlir::Type resultEleTy = resultTy.getEleTy();

  // lhs must have rank 2 for the transpose to be valid
  if ((lhsRank != 2) || ((rhsRank != 1) && (rhsRank != 2)))
    return emitOpError("array must have either rank 1 or rank 2");

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(rhsEleTy))
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

  if (mlir::isa<fir::LogicalType>(lhsEleTy) !=
      mlir::isa<fir::LogicalType>(resultEleTy))
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

  return mlir::success();
}

void hlfir::MatmulTransposeOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// AssociateOp
//===----------------------------------------------------------------------===//

void hlfir::AssociateOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &result, mlir::Value source,
                               llvm::StringRef uniq_name, mlir::Value shape,
                               mlir::ValueRange typeparams,
                               fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  mlir::Type dataType = getFortranElementOrSequenceType(source.getType());

  // Preserve polymorphism of polymorphic expr.
  mlir::Type firVarType;
  auto sourceExprType = mlir::dyn_cast<hlfir::ExprType>(source.getType());
  if (sourceExprType && sourceExprType.isPolymorphic())
    firVarType = fir::ClassType::get(fir::HeapType::get(dataType));
  else
    firVarType = fir::ReferenceType::get(dataType);

  mlir::Type hlfirVariableType =
      DeclareOp::getHLFIRVariableType(firVarType, /*hasExplicitLbs=*/false);
  mlir::Type i1Type = builder.getI1Type();
  build(builder, result, {hlfirVariableType, firVarType, i1Type}, source, shape,
        typeparams, nameAttr, fortran_attrs);
}

//===----------------------------------------------------------------------===//
// EndAssociateOp
//===----------------------------------------------------------------------===//

void hlfir::EndAssociateOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &result,
                                  hlfir::AssociateOp associate) {
  mlir::Value hlfirBase = associate.getBase();
  mlir::Value firBase = associate.getFirBase();
  // If EndAssociateOp may need to initiate the deallocation
  // of allocatable components, it has to have access to the variable
  // definition, so we cannot use the FIR base as the operand.
  return build(builder, result,
               hlfir::mayHaveAllocatableComponent(hlfirBase.getType())
                   ? hlfirBase
                   : firBase,
               associate.getMustFreeStrorageFlag());
}

mlir::LogicalResult hlfir::EndAssociateOp::verify() {
  mlir::Value var = getVar();
  if (hlfir::mayHaveAllocatableComponent(var.getType()) &&
      !hlfir::isFortranEntity(var))
    return emitOpError("that requires components deallocation must have var "
                       "operand that is a Fortran entity");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AsExprOp
//===----------------------------------------------------------------------===//

void hlfir::AsExprOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Value var,
                            mlir::Value mustFree) {
  hlfir::ExprType::Shape typeShape;
  bool isPolymorphic = fir::isPolymorphicType(var.getType());
  mlir::Type type = getFortranElementOrSequenceType(var.getType());
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    typeShape.append(seqType.getShape().begin(), seqType.getShape().end());
    type = seqType.getEleTy();
  }

  auto resultType = hlfir::ExprType::get(builder.getContext(), typeShape, type,
                                         isPolymorphic);
  return build(builder, result, resultType, var, mustFree);
}

void hlfir::AsExprOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
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

void hlfir::ElementalOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &odsState,
                               mlir::Type resultType, mlir::Value shape,
                               mlir::Value mold, mlir::ValueRange typeparams,
                               bool isUnordered) {
  odsState.addOperands(shape);
  if (mold)
    odsState.addOperands(mold);
  odsState.addOperands(typeparams);
  odsState.addTypes(resultType);
  odsState.addAttribute(
      getOperandSegmentSizesAttrName(odsState.name),
      builder.getDenseI32ArrayAttr({/*shape=*/1, (mold ? 1 : 0),
                                    static_cast<int32_t>(typeparams.size())}));
  if (isUnordered)
    odsState.addAttribute(getUnorderedAttrName(odsState.name),
                          isUnordered ? builder.getUnitAttr() : nullptr);
  mlir::Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (auto exprType = resultType.dyn_cast<hlfir::ExprType>()) {
    unsigned dim = exprType.getRank();
    mlir::Type indexType = builder.getIndexType();
    for (unsigned d = 0; d < dim; ++d)
      bodyRegion->front().addArgument(indexType, odsState.location);
  }
}

mlir::Value hlfir::ElementalOp::getElementEntity() {
  return mlir::cast<hlfir::YieldElementOp>(getBody()->back()).getElementValue();
}

mlir::LogicalResult hlfir::ElementalOp::verify() {
  mlir::Value mold = getMold();
  hlfir::ExprType resultType = mlir::cast<hlfir::ExprType>(getType());
  if (!!mold != resultType.isPolymorphic())
    return emitOpError("result must be polymorphic when mold is present "
                       "and vice versa");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

void hlfir::ApplyOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &odsState, mlir::Value expr,
                           mlir::ValueRange indices,
                           mlir::ValueRange typeparams) {
  mlir::Type resultType = expr.getType();
  if (auto exprType = resultType.dyn_cast<hlfir::ExprType>())
    resultType = exprType.getElementExprType();
  build(builder, odsState, resultType, expr, indices, typeparams);
}

//===----------------------------------------------------------------------===//
// NullOp
//===----------------------------------------------------------------------===//

void hlfir::NullOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &odsState) {
  return build(builder, odsState,
               fir::ReferenceType::get(builder.getNoneType()));
}

//===----------------------------------------------------------------------===//
// DestroyOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::DestroyOp::verify() {
  if (mustFinalizeExpr()) {
    mlir::Value expr = getExpr();
    hlfir::ExprType exprTy = mlir::cast<hlfir::ExprType>(expr.getType());
    mlir::Type elemTy = hlfir::getFortranElementType(exprTy);
    if (!mlir::isa<fir::RecordType>(elemTy))
      return emitOpError(
          "the element type must be finalizable, when 'finalize' is set");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CopyInOp
//===----------------------------------------------------------------------===//

void hlfir::CopyInOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &odsState, mlir::Value var,
                            mlir::Value var_is_present) {
  return build(builder, odsState, {var.getType(), builder.getI1Type()}, var,
               var_is_present);
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

void hlfir::ShapeOfOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value expr) {
  hlfir::ExprType exprTy = expr.getType().cast<hlfir::ExprType>();
  mlir::Type type = fir::ShapeType::get(builder.getContext(), exprTy.getRank());
  build(builder, result, type, expr);
}

std::size_t hlfir::ShapeOfOp::getRank() {
  mlir::Type resTy = getResult().getType();
  fir::ShapeType shape = resTy.cast<fir::ShapeType>();
  return shape.getRank();
}

mlir::LogicalResult hlfir::ShapeOfOp::verify() {
  mlir::Value expr = getExpr();
  hlfir::ExprType exprTy = expr.getType().cast<hlfir::ExprType>();
  std::size_t exprRank = exprTy.getShape().size();

  if (exprRank == 0)
    return emitOpError("cannot get the shape of a shape-less expression");

  std::size_t shapeRank = getRank();
  if (shapeRank != exprRank)
    return emitOpError("result rank and expr rank do not match");

  return mlir::success();
}

mlir::LogicalResult
hlfir::ShapeOfOp::canonicalize(ShapeOfOp shapeOf,
                               mlir::PatternRewriter &rewriter) {
  // if extent information is available at compile time, immediately fold the
  // hlfir.shape_of into a fir.shape
  mlir::Location loc = shapeOf.getLoc();
  hlfir::ExprType expr = shapeOf.getExpr().getType().cast<hlfir::ExprType>();

  mlir::Value shape = hlfir::genExprShape(rewriter, loc, expr);
  if (!shape)
    // shape information is not available at compile time
    return mlir::LogicalResult::failure();

  rewriter.replaceAllUsesWith(shapeOf.getResult(), shape);
  rewriter.eraseOp(shapeOf);
  return mlir::LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// GetExtent
//===----------------------------------------------------------------------===//

void hlfir::GetExtentOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &result, mlir::Value shape,
                               unsigned dim) {
  mlir::Type indexTy = builder.getIndexType();
  mlir::IntegerAttr dimAttr = mlir::IntegerAttr::get(indexTy, dim);
  build(builder, result, indexTy, shape, dimAttr);
}

mlir::LogicalResult hlfir::GetExtentOp::verify() {
  fir::ShapeType shapeTy = getShape().getType().cast<fir::ShapeType>();
  std::uint64_t rank = shapeTy.getRank();
  llvm::APInt dim = getDim();
  if (dim.sge(rank))
    return emitOpError("dimension index out of bounds");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionAssignOp
//===----------------------------------------------------------------------===//

/// Add a fir.end terminator to a parsed region if it does not already has a
/// terminator.
static void ensureTerminator(mlir::Region &region, mlir::Builder &builder,
                             mlir::Location loc) {
  // Borrow YielOp::ensureTerminator MLIR generated implementation to add a
  // fir.end if there is no terminator. This has nothing to do with YielOp,
  // other than the fact that yieldOp has the
  // SingleBlocklicitTerminator<"fir::FirEndOp"> interface that
  // cannot be added on other HLFIR operations with several regions which are
  // not all terminated the same way.
  hlfir::YieldOp::ensureTerminator(region, builder, loc);
}

mlir::ParseResult hlfir::RegionAssignOp::parse(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  mlir::Region &rhsRegion = *result.addRegion();
  if (parser.parseRegion(rhsRegion))
    return mlir::failure();
  mlir::Region &lhsRegion = *result.addRegion();
  if (parser.parseKeyword("to") || parser.parseRegion(lhsRegion))
    return mlir::failure();
  mlir::Region &userDefinedAssignmentRegion = *result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("user_defined_assign"))) {
    mlir::OpAsmParser::Argument rhsArg, lhsArg;
    if (parser.parseLParen() || parser.parseArgument(rhsArg) ||
        parser.parseColon() || parser.parseType(rhsArg.type) ||
        parser.parseRParen() || parser.parseKeyword("to") ||
        parser.parseLParen() || parser.parseArgument(lhsArg) ||
        parser.parseColon() || parser.parseType(lhsArg.type) ||
        parser.parseRParen())
      return mlir::failure();
    if (parser.parseRegion(userDefinedAssignmentRegion, {rhsArg, lhsArg}))
      return mlir::failure();
    ensureTerminator(userDefinedAssignmentRegion, parser.getBuilder(),
                     result.location);
  }
  return mlir::success();
}

void hlfir::RegionAssignOp::print(mlir::OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRhsRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p << " to ";
  p.printRegion(getLhsRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (!getUserDefinedAssignment().empty()) {
    p << " user_defined_assign ";
    mlir::Value userAssignmentRhs = getUserAssignmentRhs();
    mlir::Value userAssignmentLhs = getUserAssignmentLhs();
    p << " (" << userAssignmentRhs << ": " << userAssignmentRhs.getType()
      << ") to (";
    p << userAssignmentLhs << ": " << userAssignmentLhs.getType() << ") ";
    p.printRegion(getUserDefinedAssignment(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

static mlir::Operation *getTerminator(mlir::Region &region) {
  if (region.empty() || region.back().empty())
    return nullptr;
  return &region.back().back();
}

mlir::LogicalResult hlfir::RegionAssignOp::verify() {
  if (!mlir::isa_and_nonnull<hlfir::YieldOp>(getTerminator(getRhsRegion())))
    return emitOpError(
        "right-hand side region must be terminated by an hlfir.yield");
  if (!mlir::isa_and_nonnull<hlfir::YieldOp, hlfir::ElementalAddrOp>(
          getTerminator(getLhsRegion())))
    return emitOpError("left-hand side region must be terminated by an "
                       "hlfir.yield or hlfir.elemental_addr");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseYieldOpCleanup(mlir::OpAsmParser &parser,
                                             mlir::Region &cleanup) {
  if (succeeded(parser.parseOptionalKeyword("cleanup"))) {
    if (parser.parseRegion(cleanup, /*arguments=*/{},
                           /*argTypes=*/{}))
      return mlir::failure();
    hlfir::YieldOp::ensureTerminator(cleanup, parser.getBuilder(),
                                     parser.getBuilder().getUnknownLoc());
  }
  return mlir::success();
}

template <typename YieldOp>
static void printYieldOpCleanup(mlir::OpAsmPrinter &p, YieldOp yieldOp,
                                mlir::Region &cleanup) {
  if (!cleanup.empty()) {
    p << "cleanup ";
    p.printRegion(cleanup, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

//===----------------------------------------------------------------------===//
// ElementalAddrOp
//===----------------------------------------------------------------------===//

void hlfir::ElementalAddrOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &odsState,
                                   mlir::Value shape, bool isUnordered) {
  odsState.addOperands(shape);
  if (isUnordered)
    odsState.addAttribute(getUnorderedAttrName(odsState.name),
                          isUnordered ? builder.getUnitAttr() : nullptr);
  mlir::Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (auto shapeType = shape.getType().dyn_cast<fir::ShapeType>()) {
    unsigned dim = shapeType.getRank();
    mlir::Type indexType = builder.getIndexType();
    for (unsigned d = 0; d < dim; ++d)
      bodyRegion->front().addArgument(indexType, odsState.location);
  }
  // Push cleanUp region.
  odsState.addRegion();
}

mlir::LogicalResult hlfir::ElementalAddrOp::verify() {
  hlfir::YieldOp yieldOp =
      mlir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(getBody()));
  if (!yieldOp)
    return emitOpError("body region must be terminated by an hlfir.yield");
  mlir::Type elementAddrType = yieldOp.getEntity().getType();
  if (!hlfir::isFortranVariableType(elementAddrType) ||
      hlfir::getFortranElementOrSequenceType(elementAddrType)
          .isa<fir::SequenceType>())
    return emitOpError("body must compute the address of a scalar entity");
  unsigned shapeRank = getShape().getType().cast<fir::ShapeType>().getRank();
  if (shapeRank != getIndices().size())
    return emitOpError("body number of indices must match shape rank");
  return mlir::success();
}

hlfir::YieldOp hlfir::ElementalAddrOp::getYieldOp() {
  hlfir::YieldOp yieldOp =
      mlir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(getBody()));
  assert(yieldOp && "element_addr is ill-formed");
  return yieldOp;
}

mlir::Value hlfir::ElementalAddrOp::getElementEntity() {
  return getYieldOp().getEntity();
}

mlir::Region *hlfir::ElementalAddrOp::getElementCleanup() {
  mlir::Region *cleanup = &getYieldOp().getCleanup();
  return cleanup->empty() ? nullptr : cleanup;
}

//===----------------------------------------------------------------------===//
// OrderedAssignmentTreeOpInterface
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::OrderedAssignmentTreeOpInterface::verifyImpl() {
  if (mlir::Region *body = getSubTreeRegion())
    if (!body->empty())
      for (mlir::Operation &op : body->front())
        if (!mlir::isa<hlfir::OrderedAssignmentTreeOpInterface, fir::FirEndOp>(
                op))
          return emitOpError(
              "body region must only contain OrderedAssignmentTreeOpInterface "
              "operations or fir.end");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseForallOpBody(mlir::OpAsmParser &parser,
                                           mlir::Region &body) {
  mlir::OpAsmParser::Argument bodyArg;
  if (parser.parseLParen() || parser.parseArgument(bodyArg) ||
      parser.parseColon() || parser.parseType(bodyArg.type) ||
      parser.parseRParen())
    return mlir::failure();
  if (parser.parseRegion(body, {bodyArg}))
    return mlir::failure();
  ensureTerminator(body, parser.getBuilder(),
                   parser.getBuilder().getUnknownLoc());
  return mlir::success();
}

static void printForallOpBody(mlir::OpAsmPrinter &p, hlfir::ForallOp forall,
                              mlir::Region &body) {
  mlir::Value forallIndex = forall.getForallIndexValue();
  p << " (" << forallIndex << ": " << forallIndex.getType() << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

/// Predicate implementation of YieldIntegerOrEmpty.
static bool yieldsIntegerOrEmpty(mlir::Region &region) {
  if (region.empty())
    return true;
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(region));
  return yield && fir::isa_integer(yield.getEntity().getType());
}

//===----------------------------------------------------------------------===//
// ForallMaskOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseAssignmentMaskOpBody(mlir::OpAsmParser &parser,
                                                   mlir::Region &body) {
  if (parser.parseRegion(body))
    return mlir::failure();
  ensureTerminator(body, parser.getBuilder(),
                   parser.getBuilder().getUnknownLoc());
  return mlir::success();
}

template <typename ConcreteOp>
static void printAssignmentMaskOpBody(mlir::OpAsmPrinter &p, ConcreteOp,
                                      mlir::Region &body) {
  // ElseWhereOp is a WhereOp/ElseWhereOp terminator that should be printed.
  bool printBlockTerminators =
      !body.empty() &&
      mlir::isa_and_nonnull<hlfir::ElseWhereOp>(body.back().getTerminator());
  p.printRegion(body, /*printEntryBlockArgs=*/false, printBlockTerminators);
}

static bool yieldsLogical(mlir::Region &region, bool mustBeScalarI1) {
  if (region.empty())
    return false;
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldOp>(getTerminator(region));
  if (!yield)
    return false;
  mlir::Type yieldType = yield.getEntity().getType();
  if (mustBeScalarI1)
    return hlfir::isI1Type(yieldType);
  return hlfir::isMaskArgument(yieldType) &&
         hlfir::getFortranElementOrSequenceType(yieldType)
             .isa<fir::SequenceType>();
}

mlir::LogicalResult hlfir::ForallMaskOp::verify() {
  if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/true))
    return emitOpError("mask region must yield a scalar i1");
  mlir::Operation *op = getOperation();
  hlfir::ForallOp forallOp =
      mlir::dyn_cast_or_null<hlfir::ForallOp>(op->getParentOp());
  if (!forallOp || op->getParentRegion() != &forallOp.getBody())
    return emitOpError("must be inside the body region of an hlfir.forall");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// WhereOp and ElseWhereOp
//===----------------------------------------------------------------------===//

template <typename ConcreteOp>
static mlir::LogicalResult verifyWhereAndElseWhereBody(ConcreteOp &concreteOp) {
  for (mlir::Operation &op : concreteOp.getBody().front())
    if (mlir::isa<hlfir::ForallOp>(op))
      return concreteOp.emitOpError(
          "body region must not contain hlfir.forall");
  return mlir::success();
}

mlir::LogicalResult hlfir::WhereOp::verify() {
  if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/false))
    return emitOpError("mask region must yield a logical array");
  return verifyWhereAndElseWhereBody(*this);
}

mlir::LogicalResult hlfir::ElseWhereOp::verify() {
  if (!getMaskRegion().empty())
    if (!yieldsLogical(getMaskRegion(), /*mustBeScalarI1=*/false))
      return emitOpError(
          "mask region must yield a logical array when provided");
  return verifyWhereAndElseWhereBody(*this);
}

//===----------------------------------------------------------------------===//
// ForallIndexOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult
hlfir::ForallIndexOp::canonicalize(hlfir::ForallIndexOp indexOp,
                                   mlir::PatternRewriter &rewriter) {
  for (mlir::Operation *user : indexOp->getResult(0).getUsers())
    if (!mlir::isa<fir::LoadOp>(user))
      return mlir::failure();

  auto insertPt = rewriter.saveInsertionPoint();
  for (mlir::Operation *user : indexOp->getResult(0).getUsers())
    if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(user)) {
      rewriter.setInsertionPoint(loadOp);
      rewriter.replaceOpWithNewOp<fir::ConvertOp>(
          user, loadOp.getResult().getType(), indexOp.getIndex());
    }
  rewriter.restoreInsertionPoint(insertPt);
  rewriter.eraseOp(indexOp);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CharExtremumOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::CharExtremumOp::verify() {
  if (getStrings().size() < 2)
    return emitOpError("must be provided at least two string operands");
  unsigned kind = getCharacterKind(getResult().getType());
  for (auto string : getStrings())
    if (kind != getCharacterKind(string.getType()))
      return emitOpError("strings must have the same KIND as the result type");
  return mlir::success();
}

void hlfir::CharExtremumOp::build(mlir::OpBuilder &builder,
                                  mlir::OperationState &result,
                                  hlfir::CharExtremumPredicate predicate,
                                  mlir::ValueRange strings) {

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
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  getIntrinsicEffects(getOperation(), effects);
}

//===----------------------------------------------------------------------===//
// GetLength
//===----------------------------------------------------------------------===//

mlir::LogicalResult
hlfir::GetLengthOp::canonicalize(GetLengthOp getLength,
                                 mlir::PatternRewriter &rewriter) {
  mlir::Location loc = getLength.getLoc();
  auto exprTy = mlir::cast<hlfir::ExprType>(getLength.getExpr().getType());
  auto charTy = mlir::cast<fir::CharacterType>(exprTy.getElementType());
  if (!charTy.hasConstantLen())
    return mlir::failure();

  mlir::Type indexTy = rewriter.getIndexType();
  auto cstLen = rewriter.create<mlir::arith::ConstantOp>(
      loc, indexTy, mlir::IntegerAttr::get(indexTy, charTy.getLen()));
  rewriter.replaceOp(getLength, cstLen);
  return mlir::success();
}

#include "flang/Optimizer/HLFIR/HLFIROpInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIREnums.cpp.inc"
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
