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
#include <optional>
#include <tuple>

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

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult hlfir::SumOp::verify() {
  mlir::Operation *op = getOperation();

  auto results = op->getResultTypes();
  assert(results.size() == 1);

  mlir::Value array = getArray();
  mlir::Value mask = getMask();

  fir::SequenceType arrayTy =
      hlfir::getFortranElementOrSequenceType(array.getType())
          .cast<fir::SequenceType>();
  mlir::Type numTy = arrayTy.getEleTy();
  llvm::ArrayRef<int64_t> arrayShape = arrayTy.getShape();
  hlfir::ExprType resultTy = results[0].cast<hlfir::ExprType>();

  if (mask) {
    fir::SequenceType maskSeq =
        hlfir::getFortranElementOrSequenceType(mask.getType())
            .dyn_cast<fir::SequenceType>();
    llvm::ArrayRef<int64_t> maskShape;

    if (maskSeq)
      maskShape = maskSeq.getShape();

    if (!maskShape.empty()) {
      if (maskShape.size() != arrayShape.size())
        return emitWarning("MASK must be conformable to ARRAY");
      static_assert(fir::SequenceType::getUnknownExtent() ==
                    hlfir::ExprType::getUnknownExtent());
      constexpr int64_t unknownExtent = fir::SequenceType::getUnknownExtent();
      for (std::size_t i = 0; i < arrayShape.size(); ++i) {
        int64_t arrayExtent = arrayShape[i];
        int64_t maskExtent = maskShape[i];
        if ((arrayExtent != maskExtent) && (arrayExtent != unknownExtent) &&
            (maskExtent != unknownExtent))
          return emitWarning("MASK must be conformable to ARRAY");
      }
    }
  }

  if (resultTy.isArray()) {
    // Result is of the same type as ARRAY
    if (resultTy.getEleTy() != numTy)
      return emitOpError(
          "result must have the same element type as ARRAY argument");

    llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();

    // Result has rank n-1
    if (resultShape.size() != (arrayShape.size() - 1))
      return emitOpError("result rank must be one less than ARRAY");
  } else {
    // Result is of the same type as ARRAY
    if (resultTy.getElementType() != numTy)
      return emitOpError(
          "result must have the same element type as ARRAY argument");
  }

  return mlir::success();
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
  if (resultShape[0] != expectedResultShape[0])
    return emitOpError("incorrect result shape");
  if (resultShape.size() == 2 && resultShape[1] != expectedResultShape[1])
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

  if (inShape[0] != resultShape[1] || inShape[1] != resultShape[0])
    return emitOpError("output shape does not match input array");

  if (eleTy != resultEleTy)
    return emitOpError(
        "input and output arrays should have the same element type");

  return mlir::success();
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

//===----------------------------------------------------------------------===//
// AssociateOp
//===----------------------------------------------------------------------===//

void hlfir::AssociateOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &result, mlir::Value source,
                               llvm::StringRef uniq_name, mlir::Value shape,
                               mlir::ValueRange typeparams,
                               fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  // TODO: preserve polymorphism of polymorphic expr.
  mlir::Type firVarType = fir::ReferenceType::get(
      getFortranElementOrSequenceType(source.getType()));
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
  return build(builder, result, associate.getFirBase(),
               associate.getMustFreeStrorageFlag());
}

//===----------------------------------------------------------------------===//
// AsExprOp
//===----------------------------------------------------------------------===//

void hlfir::AsExprOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Value var,
                            mlir::Value mustFree) {
  hlfir::ExprType::Shape typeShape;
  mlir::Type type = getFortranElementOrSequenceType(var.getType());
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    typeShape.append(seqType.getShape().begin(), seqType.getShape().end());
    type = seqType.getEleTy();
  }

  auto resultType = hlfir::ExprType::get(builder.getContext(), typeShape, type,
                                         /*isPolymorphic: TODO*/ false);
  return build(builder, result, resultType, var, mustFree);
}

//===----------------------------------------------------------------------===//
// ElementalOp
//===----------------------------------------------------------------------===//

void hlfir::ElementalOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &odsState,
                               mlir::Type resultType, mlir::Value shape,
                               mlir::ValueRange typeparams) {
  odsState.addOperands(shape);
  odsState.addOperands(typeparams);
  odsState.addTypes(resultType);
  mlir::Region *bodyRegion = odsState.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (auto exprType = resultType.dyn_cast<hlfir::ExprType>()) {
    unsigned dim = exprType.getRank();
    mlir::Type indexType = builder.getIndexType();
    for (unsigned d = 0; d < dim; ++d)
      bodyRegion->front().addArgument(indexType, odsState.location);
  }
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

#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
