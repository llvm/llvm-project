//===-- FIROps.cpp --------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace fir;

/// Return true if a sequence type is of some incomplete size or a record type
/// is malformed or contains an incomplete sequence type. An incomplete sequence
/// type is one with more unknown extents in the type than have been provided
/// via `dynamicExtents`. Sequence types with an unknown rank are incomplete by
/// definition.
static bool verifyInType(mlir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited,
                         unsigned dynamicExtents = 0) {
  if (auto st = inType.dyn_cast<fir::SequenceType>()) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (std::size_t i = 0, end{shape.size()}; i < end; ++i) {
      if (shape[i] != fir::SequenceType::getUnknownExtent())
        continue;
      if (dynamicExtents-- == 0)
        return true;
    }
  } else if (auto rt = inType.dyn_cast<fir::RecordType>()) {
    // don't recurse if we're already visiting this one
    if (llvm::is_contained(visited, rt.getName()))
      return false;
    // keep track of record types currently being visited
    visited.push_back(rt.getName());
    for (auto &field : rt.getTypeList())
      if (verifyInType(field.second, visited))
        return true;
    visited.pop_back();
  } else if (auto rt = inType.dyn_cast<fir::PointerType>()) {
    return verifyInType(rt.getEleTy(), visited);
  }
  return false;
}

static bool verifyTypeParamCount(mlir::Type inType, unsigned numParams) {
  auto ty = fir::unwrapSequenceType(inType);
  if (numParams > 0) {
    if (auto recTy = ty.dyn_cast<fir::RecordType>())
      return numParams != recTy.getNumLenParams();
    if (auto chrTy = ty.dyn_cast<fir::CharacterType>())
      return !(numParams == 1 && chrTy.hasDynamicLen());
    return true;
  }
  if (auto chrTy = ty.dyn_cast<fir::CharacterType>())
    return !chrTy.hasConstantLen();
  return false;
}

/// Parser shared by Alloca and Allocmem
///
/// operation ::= %res = (`fir.alloca` | `fir.allocmem`) $in_type
///                      ( `(` $typeparams `)` )? ( `,` $shape )?
///                      attr-dict-without-keyword
template <typename FN>
static mlir::ParseResult parseAllocatableOp(FN wrapResultType,
                                            mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  llvm::SmallVector<mlir::OpAsmParser::OperandType> operands;
  llvm::SmallVector<mlir::Type> typeVec;
  bool hasOperands = false;
  std::int32_t typeparamsSize = 0;
  if (!parser.parseOptionalLParen()) {
    // parse the LEN params of the derived type. (<params> : <types>)
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(typeVec) || parser.parseRParen())
      return mlir::failure();
    typeparamsSize = operands.size();
    hasOperands = true;
  }
  std::int32_t shapeSize = 0;
  if (!parser.parseOptionalComma()) {
    // parse size to scale by, vector of n dimensions of type index
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None))
      return mlir::failure();
    shapeSize = operands.size() - typeparamsSize;
    auto idxTy = builder.getIndexType();
    for (std::int32_t i = typeparamsSize, end = operands.size(); i != end; ++i)
      typeVec.push_back(idxTy);
    hasOperands = true;
  }
  if (hasOperands &&
      parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                             result.operands))
    return mlir::failure();
  mlir::Type restype = wrapResultType(intype);
  if (!restype) {
    parser.emitError(parser.getNameLoc(), "invalid allocate type: ") << intype;
    return mlir::failure();
  }
  result.addAttribute("operand_segment_sizes",
                      builder.getI32VectorAttr({typeparamsSize, shapeSize}));
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

template <typename OP>
static void printAllocatableOp(mlir::OpAsmPrinter &p, OP &op) {
  p << ' ' << op.in_type();
  if (!op.typeparams().empty()) {
    p << '(' << op.typeparams() << " : " << op.typeparams().getTypes() << ')';
  }
  // print the shape of the allocation (if any); all must be index type
  for (auto sh : op.shape()) {
    p << ", ";
    p.printOperand(sh);
  }
  p.printOptionalAttrDict(op->getAttrs(), {"in_type", "operand_segment_sizes"});
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Create a legal memory reference as return type
static mlir::Type wrapAllocaResultType(mlir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (intype.isa<ReferenceType>())
    return {};
  return ReferenceType::get(intype);
}

mlir::Type fir::AllocaOp::getAllocatedType() {
  return getType().cast<ReferenceType>().getEleTy();
}

mlir::Type fir::AllocaOp::getRefTy(mlir::Type ty) {
  return ReferenceType::get(ty);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, bool pinned,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? mlir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? mlir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          bool pinned, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? mlir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? mlir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          mlir::ValueRange typeparams, mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Type inType,
                          bool pinned, mlir::ValueRange typeparams,
                          mlir::ValueRange shape,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {}, pinned,
        typeparams, shape);
  result.addAttributes(attributes);
}

static mlir::LogicalResult verify(fir::AllocaOp &op) {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(op.getInType(), visited, op.numShapeOperands()))
    return op.emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(op.getInType(), op.numLenParams()))
    return op.emitOpError("LEN params do not correspond to type");
  mlir::Type outType = op.getType();
  if (!outType.isa<fir::ReferenceType>())
    return op.emitOpError("must be a !fir.ref type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return op.emitOpError("cannot allocate !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AllocMemOp
//===----------------------------------------------------------------------===//

/// Create a legal heap reference as return type
static mlir::Type wrapAllocMemResultType(mlir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (intype.isa<ReferenceType>() || intype.isa<HeapType>() ||
      intype.isa<PointerType>() || intype.isa<FunctionType>())
    return {};
  return HeapType::get(intype);
}

mlir::Type fir::AllocMemOp::getAllocatedType() {
  return getType().cast<HeapType>().getEleTy();
}

mlir::Type fir::AllocMemOp::getRefTy(mlir::Type ty) {
  return HeapType::get(ty);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            llvm::StringRef uniqName,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            llvm::StringRef uniqName, llvm::StringRef bindcName,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  auto bindcAttr = builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr,
        bindcAttr, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, mlir::Type inType,
                            mlir::ValueRange typeparams, mlir::ValueRange shape,
                            llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocMemResultType(inType), inType, {}, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

static mlir::LogicalResult verify(fir::AllocMemOp op) {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(op.getInType(), visited, op.numShapeOperands()))
    return op.emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(op.getInType(), op.numLenParams()))
    return op.emitOpError("LEN params do not correspond to type");
  mlir::Type outType = op.getType();
  if (!outType.dyn_cast<fir::HeapType>())
    return op.emitOpError("must be a !fir.heap type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return op.emitOpError("cannot allocate !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayCoorOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ArrayCoorOp op) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return op.emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = op.shape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!op.memref().getType().isa<fir::BoxType>())
        return op.emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return op.emitOpError("rank of dimension mismatched");
    if (shapeTyRank != op.indices().size())
      return op.emitOpError("number of indices do not match dim rank");
  }

  if (auto sliceOp = op.slice())
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return op.emitOpError("rank of dimension in slice mismatched");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayLoadOp
//===----------------------------------------------------------------------===//

static mlir::Type adjustedElementType(mlir::Type t) {
  if (auto ty = t.dyn_cast<fir::ReferenceType>()) {
    auto eleTy = ty.getEleTy();
    if (fir::isa_char(eleTy))
      return eleTy;
    if (fir::isa_derived(eleTy))
      return eleTy;
    if (eleTy.isa<fir::SequenceType>())
      return eleTy;
  }
  return t;
}

std::vector<mlir::Value> fir::ArrayLoadOp::getExtents() {
  if (auto sh = shape())
    if (auto *op = sh.getDefiningOp()) {
      if (auto shOp = dyn_cast<fir::ShapeOp>(op))
        return shOp.getExtents();
      return cast<fir::ShapeShiftOp>(op).getExtents();
    }
  return {};
}

static mlir::LogicalResult verify(fir::ArrayLoadOp op) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return op.emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = op.shape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!op.memref().getType().isa<fir::BoxType>())
        return op.emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return op.emitOpError("rank of dimension mismatched");
  }

  if (auto sliceOp = op.slice())
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return op.emitOpError("rank of dimension in slice mismatched");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayMergeStoreOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ArrayMergeStoreOp op) {
  if (!isa<ArrayLoadOp>(op.original().getDefiningOp()))
    return op.emitOpError("operand #0 must be result of a fir.array_load op");
  if (auto sl = op.slice()) {
    if (auto *slOp = sl.getDefiningOp()) {
      auto sliceOp = mlir::cast<fir::SliceOp>(slOp);
      if (!sliceOp.fields().empty()) {
        // This is an intra-object merge, where the slice is projecting the
        // subfields that are to be overwritten by the merge operation.
        auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
        if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
          auto projTy =
              fir::applyPathToType(seqTy.getEleTy(), sliceOp.fields());
          if (fir::unwrapSequenceType(op.original().getType()) != projTy)
            return op.emitOpError(
                "type of origin does not match sliced memref type");
          if (fir::unwrapSequenceType(op.sequence().getType()) != projTy)
            return op.emitOpError(
                "type of sequence does not match sliced memref type");
          return mlir::success();
        }
        return op.emitOpError("referenced type is not an array");
      }
    }
    return mlir::success();
  }
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
  if (op.original().getType() != eleTy)
    return op.emitOpError("type of origin does not match memref element type");
  if (op.sequence().getType() != eleTy)
    return op.emitOpError(
        "type of sequence does not match memref element type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayFetchOp
//===----------------------------------------------------------------------===//

// Template function used for both array_fetch and array_update verification.
template <typename A>
mlir::Type validArraySubobject(A op) {
  auto ty = op.sequence().getType();
  return fir::applyPathToType(ty, op.indices());
}

static mlir::LogicalResult verify(fir::ArrayFetchOp op) {
  auto arrTy = op.sequence().getType().cast<fir::SequenceType>();
  auto indSize = op.indices().size();
  if (indSize < arrTy.getDimension())
    return op.emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(op.element().getType()) != arrTy.getEleTy())
    return op.emitOpError("return type does not match array");
  auto ty = validArraySubobject(op);
  if (!ty || ty != ::adjustedElementType(op.getType()))
    return op.emitOpError("return type and/or indices do not type check");
  if (!isa<fir::ArrayLoadOp>(op.sequence().getDefiningOp()))
    return op.emitOpError("argument #0 must be result of fir.array_load");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayUpdateOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ArrayUpdateOp op) {
  auto arrTy = op.sequence().getType().cast<fir::SequenceType>();
  auto indSize = op.indices().size();
  if (indSize < arrTy.getDimension())
    return op.emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(op.merge().getType()) != arrTy.getEleTy())
    return op.emitOpError("merged value does not have element type");
  auto ty = validArraySubobject(op);
  if (!ty || ty != ::adjustedElementType(op.merge().getType()))
    return op.emitOpError("merged value and/or indices do not type check");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayModifyOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ArrayModifyOp op) {
  auto arrTy = op.sequence().getType().cast<fir::SequenceType>();
  auto indSize = op.indices().size();
  if (indSize < arrTy.getDimension())
    return op.emitOpError("number of indices must match array dimension");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxAddrOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult fir::BoxAddrOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto v = val().getDefiningOp()) {
    if (auto box = dyn_cast<fir::EmboxOp>(v))
      return box.memref();
    if (auto box = dyn_cast<fir::EmboxCharOp>(v))
      return box.memref();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxCharLenOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult
fir::BoxCharLenOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto v = val().getDefiningOp()) {
    if (auto box = dyn_cast<fir::EmboxCharOp>(v))
      return box.len();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxDimsOp
//===----------------------------------------------------------------------===//

/// Get the result types packed in a tuple tuple
mlir::Type fir::BoxDimsOp::getTupleType() {
  // note: triple, but 4 is nearest power of 2
  llvm::SmallVector<mlir::Type> triple{
      getResult(0).getType(), getResult(1).getType(), getResult(2).getType()};
  return mlir::TupleType::get(getContext(), triple);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::CallOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

static void printCallOp(mlir::OpAsmPrinter &p, fir::CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();
  p << ' ';
  if (isDirect)
    p << callee.getValue();
  else
    p << op.getOperand(0);
  p << '(' << op->getOperands().drop_front(isDirect ? 0 : 1) << ')';
  p.printOptionalAttrDict(op->getAttrs(), {"callee"});
  auto resultTypes{op.getResultTypes()};
  llvm::SmallVector<Type> argTypes(
      llvm::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << FunctionType::get(op.getContext(), argTypes, resultTypes);
}

static mlir::ParseResult parseCallOp(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType> operands;
  if (parser.parseOperandList(operands))
    return mlir::failure();

  mlir::NamedAttrList attrs;
  mlir::SymbolRefAttr funcAttr;
  bool isDirect = operands.empty();
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", attrs))
      return mlir::failure();

  Type type;
  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return mlir::failure();

  auto funcType = type.dyn_cast<mlir::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<mlir::OpAsmParser::OperandType>(operands).drop_front();
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return mlir::success();
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::FuncOp callee, mlir::ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrName(), SymbolRefAttr::get(callee));
  result.addTypes(callee.getType().getResults());
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::SymbolRefAttr callee,
                        llvm::ArrayRef<mlir::Type> results,
                        mlir::ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrName(), callee);
  result.addTypes(results);
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

template <typename OPTY>
static void printCmpOp(OpAsmPrinter &p, OPTY op) {
  p << ' ';
  auto predSym = mlir::symbolizeCmpFPredicate(
      op->template getAttrOfType<mlir::IntegerAttr>(
            OPTY::getPredicateAttrName())
          .getInt());
  assert(predSym.hasValue() && "invalid symbol value for predicate");
  p << '"' << mlir::stringifyCmpFPredicate(predSym.getValue()) << '"' << ", ";
  p.printOperand(op.lhs());
  p << ", ";
  p.printOperand(op.rhs());
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.lhs().getType();
}

template <typename OPTY>
static mlir::ParseResult parseCmpOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType> ops;
  mlir::NamedAttrList attrs;
  mlir::Attribute predicateNameAttr;
  mlir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<mlir::StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      predicateNameAttr.cast<mlir::StringAttr>().getValue();
  auto predicate = fir::CmpcOp::getPredicateByName(predicateName);
  auto builder = parser.getBuilder();
  mlir::Type i1Type = builder.getI1Type();
  attrs.set(OPTY::getPredicateAttrName(),
            builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return success();
}

//===----------------------------------------------------------------------===//
// CharConvertOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::CharConvertOp op) {
  auto unwrap = [&](mlir::Type t) {
    t = fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t));
    return t.dyn_cast<fir::CharacterType>();
  };
  auto inTy = unwrap(op.from().getType());
  auto outTy = unwrap(op.to().getType());
  if (!(inTy && outTy))
    return op.emitOpError("not a reference to a character");
  if (inTy.getFKind() == outTy.getFKind())
    return op.emitOpError("buffers must have different KIND values");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CmpcOp
//===----------------------------------------------------------------------===//

void fir::buildCmpCOp(OpBuilder &builder, OperationState &result,
                      CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      fir::CmpcOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

mlir::CmpFPredicate fir::CmpcOp::getPredicateByName(llvm::StringRef name) {
  auto pred = mlir::symbolizeCmpFPredicate(name);
  assert(pred.hasValue() && "invalid predicate name");
  return pred.getValue();
}

static void printCmpcOp(OpAsmPrinter &p, fir::CmpcOp op) { printCmpOp(p, op); }

mlir::ParseResult fir::parseCmpcOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// ConstcOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseConstcOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  fir::RealAttr realp;
  fir::RealAttr imagp;
  mlir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(realp, fir::ConstcOp::realAttrName(),
                            result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(imagp, fir::ConstcOp::imagAttrName(),
                            result.attributes) ||
      parser.parseRParen() || parser.parseColonType(type) ||
      parser.addTypesToList(type, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::ConstcOp &op) {
  p << " (0x";
  auto f1 = op.getOperation()
                ->getAttr(fir::ConstcOp::realAttrName())
                .cast<mlir::FloatAttr>();
  auto i1 = f1.getValue().bitcastToAPInt();
  p.getStream().write_hex(i1.getZExtValue());
  p << ", 0x";
  auto f2 = op.getOperation()
                ->getAttr(fir::ConstcOp::imagAttrName())
                .cast<mlir::FloatAttr>();
  auto i2 = f2.getValue().bitcastToAPInt();
  p.getStream().write_hex(i2.getZExtValue());
  p << ") : ";
  p.printType(op.getType());
}

static mlir::LogicalResult verify(fir::ConstcOp &op) {
  if (!op.getType().isa<fir::ComplexType>())
    return op.emitOpError("must be a !fir.complex type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void fir::ConvertOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {}

mlir::OpFoldResult fir::ConvertOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (value().getType() == getType())
    return value();
  if (matchPattern(value(), m_Op<fir::ConvertOp>())) {
    auto inner = cast<fir::ConvertOp>(value().getDefiningOp());
    // (convert (convert 'a : logical -> i1) : i1 -> logical) ==> forward 'a
    if (auto toTy = getType().dyn_cast<fir::LogicalType>())
      if (auto fromTy = inner.value().getType().dyn_cast<fir::LogicalType>())
        if (inner.getType().isa<mlir::IntegerType>() && (toTy == fromTy))
          return inner.value();
    // (convert (convert 'a : i1 -> logical) : logical -> i1) ==> forward 'a
    if (auto toTy = getType().dyn_cast<mlir::IntegerType>())
      if (auto fromTy = inner.value().getType().dyn_cast<mlir::IntegerType>())
        if (inner.getType().isa<fir::LogicalType>() && (toTy == fromTy) &&
            (fromTy.getWidth() == 1))
          return inner.value();
  }
  return {};
}

bool fir::ConvertOp::isIntegerCompatible(mlir::Type ty) {
  return ty.isa<mlir::IntegerType>() || ty.isa<mlir::IndexType>() ||
         ty.isa<fir::IntegerType>() || ty.isa<fir::LogicalType>();
}

bool fir::ConvertOp::isFloatCompatible(mlir::Type ty) {
  return ty.isa<mlir::FloatType>() || ty.isa<fir::RealType>();
}

bool fir::ConvertOp::isPointerCompatible(mlir::Type ty) {
  return ty.isa<fir::ReferenceType>() || ty.isa<fir::PointerType>() ||
         ty.isa<fir::HeapType>() || ty.isa<mlir::MemRefType>() ||
         ty.isa<mlir::FunctionType>() || ty.isa<fir::TypeDescType>();
}

static mlir::LogicalResult verify(fir::ConvertOp &op) {
  auto inType = op.value().getType();
  auto outType = op.getType();
  if (inType == outType)
    return mlir::success();
  if ((op.isPointerCompatible(inType) && op.isPointerCompatible(outType)) ||
      (op.isIntegerCompatible(inType) && op.isIntegerCompatible(outType)) ||
      (op.isIntegerCompatible(inType) && op.isFloatCompatible(outType)) ||
      (op.isFloatCompatible(inType) && op.isIntegerCompatible(outType)) ||
      (op.isFloatCompatible(inType) && op.isFloatCompatible(outType)) ||
      (op.isIntegerCompatible(inType) && op.isPointerCompatible(outType)) ||
      (op.isPointerCompatible(inType) && op.isIntegerCompatible(outType)) ||
      (inType.isa<fir::BoxType>() && outType.isa<fir::BoxType>()) ||
      (fir::isa_complex(inType) && fir::isa_complex(outType)))
    return mlir::success();
  return op.emitOpError("invalid type conversion");
}

//===----------------------------------------------------------------------===//
// CoordinateOp
//===----------------------------------------------------------------------===//

static void print(mlir::OpAsmPrinter &p, fir::CoordinateOp op) {
  p << ' ' << op.ref() << ", " << op.coor();
  p.printOptionalAttrDict(op->getAttrs(), /*elideAttrs=*/{"baseType"});
  p << " : ";
  p.printFunctionalType(op.getOperandTypes(), op->getResultTypes());
}

static mlir::ParseResult parseCoordinateCustom(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType memref;
  if (parser.parseOperand(memref) || parser.parseComma())
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::OperandType> coorOperands;
  if (parser.parseOperandList(coorOperands))
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::OperandType> allOperands;
  allOperands.push_back(memref);
  allOperands.append(coorOperands.begin(), coorOperands.end());
  mlir::FunctionType funcTy;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcTy) ||
      parser.resolveOperands(allOperands, funcTy.getInputs(), loc,
                             result.operands))
    return failure();
  parser.addTypesToList(funcTy.getResults(), result.types);
  result.addAttribute("baseType", mlir::TypeAttr::get(funcTy.getInput(0)));
  return mlir::success();
}

static mlir::LogicalResult verify(fir::CoordinateOp op) {
  auto refTy = op.ref().getType();
  if (fir::isa_ref_type(refTy)) {
    auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
    if (auto arrTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (arrTy.hasUnknownShape())
        return op.emitOpError("cannot find coordinate in unknown shape");
      if (arrTy.getConstantRows() < arrTy.getDimension() - 1)
        return op.emitOpError("cannot find coordinate with unknown extents");
    }
    if (!(fir::isa_aggregate(eleTy) || fir::isa_complex(eleTy) ||
          fir::isa_char_string(eleTy)))
      return op.emitOpError("cannot apply coordinate_of to this type");
  }
  // Recovering a LEN type parameter only makes sense from a boxed value. For a
  // bare reference, the LEN type parameters must be passed as additional
  // arguments to `op`.
  for (auto co : op.coor())
    if (dyn_cast_or_null<fir::LenParamIndexOp>(co.getDefiningOp())) {
      if (op.getNumOperands() != 2)
        return op.emitOpError("len_param_index must be last argument");
      if (!op.ref().getType().isa<BoxType>())
        return op.emitOpError("len_param_index must be used on box type");
    }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::DispatchOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

static mlir::ParseResult parseDispatchOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::FunctionType calleeType;
  llvm::SmallVector<mlir::OpAsmParser::OperandType> operands;
  auto calleeLoc = parser.getNameLoc();
  llvm::StringRef calleeName;
  if (failed(parser.parseOptionalKeyword(&calleeName))) {
    mlir::StringAttr calleeAttr;
    if (parser.parseAttribute(calleeAttr, fir::DispatchOp::getMethodAttrName(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::DispatchOp::getMethodAttrName(),
                        parser.getBuilder().getStringAttr(calleeName));
  }
  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), result.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::DispatchOp &op) {
  p << ' ' << op.getOperation()->getAttr(fir::DispatchOp::getMethodAttrName())
    << '(';
  p.printOperand(op.object());
  if (!op.args().empty()) {
    p << ", ";
    p.printOperands(op.args());
  }
  p << ") : ";
  p.printFunctionalType(op.getOperation()->getOperandTypes(),
                        op.getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// DispatchTableOp
//===----------------------------------------------------------------------===//

void fir::DispatchTableOp::appendTableEntry(mlir::Operation *op) {
  assert(mlir::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  auto &block = getBlock();
  block.getOperations().insert(block.end(), op);
}

static mlir::ParseResult parseDispatchTableOp(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                            result.attributes))
    return failure();

  // Convert the parsed name attr into a string attr.
  result.attributes.set(mlir::SymbolTable::getSymbolAttrName(),
                        nameAttr.getRootReference());

  // Parse the optional table body.
  mlir::Region *body = result.addRegion();
  OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
  if (parseResult.hasValue() && failed(*parseResult))
    return mlir::failure();

  fir::DispatchTableOp::ensureTerminator(*body, parser.getBuilder(),
                                         result.location);
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::DispatchTableOp &op) {
  auto tableName =
      op.getOperation()
          ->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << " @" << tableName;

  Region &body = op.getOperation()->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
}

static mlir::LogicalResult verify(fir::DispatchTableOp &op) {
  for (auto &op : op.getBlock())
    if (!(isa<fir::DTEntryOp>(op) || isa<fir::FirEndOp>(op)))
      return op.emitOpError("dispatch table must contain dt_entry");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::EmboxOp op) {
  auto eleTy = fir::dyn_cast_ptrEleTy(op.memref().getType());
  bool isArray = false;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
    eleTy = seqTy.getEleTy();
    isArray = true;
  }
  if (op.hasLenParams()) {
    auto lenPs = op.numLenParams();
    if (auto rt = eleTy.dyn_cast<fir::RecordType>()) {
      if (lenPs != rt.getNumLenParams())
        return op.emitOpError("number of LEN params does not correspond"
                              " to the !fir.type type");
    } else if (auto strTy = eleTy.dyn_cast<fir::CharacterType>()) {
      if (strTy.getLen() != fir::CharacterType::unknownLen())
        return op.emitOpError("CHARACTER already has static LEN");
    } else {
      return op.emitOpError("LEN parameters require CHARACTER or derived type");
    }
    for (auto lp : op.typeparams())
      if (!fir::isa_integer(lp.getType()))
        return op.emitOpError("LEN parameters must be integral type");
  }
  if (op.getShape() && !isArray)
    return op.emitOpError("shape must not be provided for a scalar");
  if (op.getSlice() && !isArray)
    return op.emitOpError("slice must not be provided for a scalar");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxCharOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::EmboxCharOp &op) {
  auto eleTy = fir::dyn_cast_ptrEleTy(op.memref().getType());
  if (!eleTy.dyn_cast_or_null<CharacterType>())
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxProcOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseEmboxProcOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  mlir::SymbolRefAttr procRef;
  if (parser.parseAttribute(procRef, "funcname", result.attributes))
    return mlir::failure();
  bool hasTuple = false;
  mlir::OpAsmParser::OperandType tupleRef;
  if (!parser.parseOptionalComma()) {
    if (parser.parseOperand(tupleRef))
      return mlir::failure();
    hasTuple = true;
  }
  mlir::FunctionType type;
  if (parser.parseColon() || parser.parseLParen() || parser.parseType(type))
    return mlir::failure();
  result.addAttribute("functype", mlir::TypeAttr::get(type));
  if (hasTuple) {
    mlir::Type tupleType;
    if (parser.parseComma() || parser.parseType(tupleType) ||
        parser.resolveOperand(tupleRef, tupleType, result.operands))
      return mlir::failure();
  }
  mlir::Type boxType;
  if (parser.parseRParen() || parser.parseArrow() ||
      parser.parseType(boxType) || parser.addTypesToList(boxType, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::EmboxProcOp &op) {
  p << ' ' << op.getOperation()->getAttr("funcname");
  auto h = op.host();
  if (h) {
    p << ", ";
    p.printOperand(h);
  }
  p << " : (" << op.getOperation()->getAttr("functype");
  if (h)
    p << ", " << h.getType();
  p << ") -> " << op.getType();
}

static mlir::LogicalResult verify(fir::EmboxProcOp &op) {
  // host bindings (optional) must be a reference to a tuple
  if (auto h = op.host()) {
    if (auto r = h.getType().dyn_cast<ReferenceType>()) {
      if (!r.getEleTy().dyn_cast<mlir::TupleType>())
        return mlir::failure();
    } else {
      return mlir::failure();
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenTypeDescOp
//===----------------------------------------------------------------------===//

void fir::GenTypeDescOp::build(OpBuilder &, OperationState &result,
                               mlir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

static mlir::ParseResult parseGenTypeDescOp(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  mlir::Type restype = TypeDescType::get(intype);
  if (parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::GenTypeDescOp &op) {
  p << ' ' << op.getOperation()->getAttr("in_type");
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), {"in_type"});
}

static mlir::LogicalResult verify(fir::GenTypeDescOp &op) {
  mlir::Type resultTy = op.getType();
  if (auto tdesc = resultTy.dyn_cast<TypeDescType>()) {
    if (tdesc.getOfTy() != op.getInType())
      return op.emitOpError("wrapped type mismatched");
  } else {
    return op.emitOpError("must be !fir.tdesc type");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  // Parse the optional linkage
  llvm::StringRef linkage;
  auto &builder = parser.getBuilder();
  if (mlir::succeeded(parser.parseOptionalKeyword(&linkage))) {
    if (fir::GlobalOp::verifyValidLinkage(linkage))
      return mlir::failure();
    mlir::StringAttr linkAttr = builder.getStringAttr(linkage);
    result.addAttribute(fir::GlobalOp::linkageAttrName(), linkAttr);
  }

  // Parse the name as a symbol reference attribute.
  mlir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, fir::GlobalOp::symbolAttrName(),
                            result.attributes))
    return mlir::failure();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      nameAttr.getRootReference());

  bool simpleInitializer = false;
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    Attribute attr;
    if (parser.parseAttribute(attr, "initVal", result.attributes) ||
        parser.parseRParen())
      return mlir::failure();
    simpleInitializer = true;
  }

  if (succeeded(parser.parseOptionalKeyword("constant"))) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute("constant", builder.getUnitAttr());
  }

  mlir::Type globalType;
  if (parser.parseColonType(globalType))
    return mlir::failure();

  result.addAttribute(fir::GlobalOp::typeAttrName(result.name),
                      mlir::TypeAttr::get(globalType));

  if (simpleInitializer) {
    result.addRegion();
  } else {
    // Parse the optional initializer body.
    auto parseResult = parser.parseOptionalRegion(
        *result.addRegion(), /*arguments=*/llvm::None, /*argTypes=*/llvm::None);
    if (parseResult.hasValue() && mlir::failed(*parseResult))
      return mlir::failure();
  }

  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::GlobalOp &op) {
  if (op.linkName().hasValue())
    p << ' ' << op.linkName().getValue();
  p << ' ';
  p.printAttributeWithoutType(
      op.getOperation()->getAttr(fir::GlobalOp::symbolAttrName()));
  if (auto val = op.getValueOrNull())
    p << '(' << val << ')';
  if (op.getOperation()->getAttr(fir::GlobalOp::getConstantAttrName()))
    p << " constant";
  p << " : ";
  p.printType(op.getType());
  if (op.hasInitializationBody())
    p.printRegion(op.getOperation()->getRegion(0),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

void fir::GlobalOp::appendInitialValue(mlir::Operation *op) {
  getBlock().getOperations().push_back(op);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          Attribute initialVal, StringAttr linkage,
                          ArrayRef<NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(typeAttrName(result.name), mlir::TypeAttr::get(type));
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(symbolAttrName(),
                      SymbolRefAttr::get(builder.getContext(), name));
  if (isConstant)
    result.addAttribute(constantAttrName(result.name), builder.getUnitAttr());
  if (initialVal)
    result.addAttribute(initValAttrName(result.name), initialVal);
  if (linkage)
    result.addAttribute(linkageAttrName(), linkage);
  result.attributes.append(attrs.begin(), attrs.end());
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type, Attribute initialVal,
                          StringAttr linkage, ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          StringAttr linkage, ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type, StringAttr linkage,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, StringAttr{}, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, attrs);
}

mlir::ParseResult fir::GlobalOp::verifyValidLinkage(StringRef linkage) {
  // Supporting only a subset of the LLVM linkage types for now
  static const char *validNames[] = {"common", "internal", "linkonce", "weak"};
  return mlir::success(llvm::is_contained(validNames, linkage));
}

template <bool AllowFields>
static void appendAsAttribute(llvm::SmallVectorImpl<mlir::Attribute> &attrs,
                              mlir::Value val) {
  if (auto *op = val.getDefiningOp()) {
    if (auto cop = mlir::dyn_cast<mlir::ConstantOp>(op)) {
      // append the integer constant value
      if (auto iattr = cop.getValue().dyn_cast<mlir::IntegerAttr>()) {
        attrs.push_back(iattr);
        return;
      }
    } else if (auto fld = mlir::dyn_cast<fir::FieldIndexOp>(op)) {
      if constexpr (AllowFields) {
        // append the field name and the record type
        attrs.push_back(fld.field_idAttr());
        attrs.push_back(fld.on_typeAttr());
        return;
      }
    }
  }
  llvm::report_fatal_error("cannot build Op with these arguments");
}

template <bool AllowFields = true>
static mlir::ArrayAttr collectAsAttributes(mlir::MLIRContext *ctxt,
                                           OperationState &result,
                                           llvm::ArrayRef<mlir::Value> inds) {
  llvm::SmallVector<mlir::Attribute> attrs;
  for (auto v : inds)
    appendAsAttribute<AllowFields>(attrs, v);
  assert(!attrs.empty());
  return mlir::ArrayAttr::get(ctxt, attrs);
}

//===----------------------------------------------------------------------===//
// GlobalLenOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseGlobalLenOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  llvm::StringRef fieldName;
  if (failed(parser.parseOptionalKeyword(&fieldName))) {
    mlir::StringAttr fieldAttr;
    if (parser.parseAttribute(fieldAttr, fir::GlobalLenOp::lenParamAttrName(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::GlobalLenOp::lenParamAttrName(),
                        parser.getBuilder().getStringAttr(fieldName));
  }
  mlir::IntegerAttr constant;
  if (parser.parseComma() ||
      parser.parseAttribute(constant, fir::GlobalLenOp::intAttrName(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::GlobalLenOp &op) {
  p << ' ' << op.getOperation()->getAttr(fir::GlobalLenOp::lenParamAttrName())
    << ", " << op.getOperation()->getAttr(fir::GlobalLenOp::intAttrName());
}

//===----------------------------------------------------------------------===//
// ExtractValueOp
//===----------------------------------------------------------------------===//

void fir::ExtractValueOp::build(mlir::OpBuilder &builder,
                                OperationState &result, mlir::Type resTy,
                                mlir::Value aggVal,
                                llvm::ArrayRef<mlir::Value> inds) {
  auto aa = collectAsAttributes<>(builder.getContext(), result, inds);
  build(builder, result, resTy, aggVal, aa);
}

//===----------------------------------------------------------------------===//
// FieldIndexOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseFieldIndexOp(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  mlir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::fieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!recty.dyn_cast<RecordType>())
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::typeAttrName(),
                      mlir::TypeAttr::get(recty));
  if (!parser.parseOptionalLParen()) {
    llvm::SmallVector<mlir::OpAsmParser::OperandType> operands;
    llvm::SmallVector<mlir::Type> types;
    auto loc = parser.getNameLoc();
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(types) || parser.parseRParen() ||
        parser.resolveOperands(operands, types, loc, result.operands))
      return mlir::failure();
  }
  mlir::Type fieldType = fir::FieldType::get(builder.getContext());
  if (parser.addTypeToList(fieldType, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::FieldIndexOp &op) {
  p << ' '
    << op.getOperation()
           ->getAttrOfType<mlir::StringAttr>(fir::FieldIndexOp::fieldAttrName())
           .getValue()
    << ", " << op.getOperation()->getAttr(fir::FieldIndexOp::typeAttrName());
  if (op.getNumOperands()) {
    p << '(';
    p.printOperands(op.typeparams());
    const auto *sep = ") : ";
    for (auto op : op.typeparams()) {
      p << sep;
      if (op)
        p.printType(op.getType());
      else
        p << "()";
      sep = ", ";
    }
  }
}

void fir::FieldIndexOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              llvm::StringRef fieldName, mlir::Type recTy,
                              mlir::ValueRange operands) {
  result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(typeAttrName(), TypeAttr::get(recTy));
  result.addOperands(operands);
}

//===----------------------------------------------------------------------===//
// InsertOnRangeOp
//===----------------------------------------------------------------------===//

void fir::InsertOnRangeOp::build(mlir::OpBuilder &builder,
                                 OperationState &result, mlir::Type resTy,
                                 mlir::Value aggVal, mlir::Value eleVal,
                                 llvm::ArrayRef<mlir::Value> inds) {
  auto aa = collectAsAttributes<false>(builder.getContext(), result, inds);
  build(builder, result, resTy, aggVal, eleVal, aa);
}

/// Range bounds must be nonnegative, and the range must not be empty.
static mlir::LogicalResult verify(fir::InsertOnRangeOp op) {
  if (op.coor().size() < 2 || op.coor().size() % 2 != 0)
    return op.emitOpError("has uneven number of values in ranges");
  bool rangeIsKnownToBeNonempty = false;
  for (auto i = op.coor().end(), b = op.coor().begin(); i != b;) {
    int64_t ub = (*--i).cast<IntegerAttr>().getInt();
    int64_t lb = (*--i).cast<IntegerAttr>().getInt();
    if (lb < 0 || ub < 0)
      return op.emitOpError("negative range bound");
    if (rangeIsKnownToBeNonempty)
      continue;
    if (lb > ub)
      return op.emitOpError("empty range");
    rangeIsKnownToBeNonempty = lb < ub;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

void fir::InsertValueOp::build(mlir::OpBuilder &builder, OperationState &result,
                               mlir::Type resTy, mlir::Value aggVal,
                               mlir::Value eleVal,
                               llvm::ArrayRef<mlir::Value> inds) {
  auto aa = collectAsAttributes<>(builder.getContext(), result, inds);
  build(builder, result, resTy, aggVal, eleVal, aa);
}

static bool checkIsIntegerConstant(mlir::Attribute attr, int64_t conVal) {
  if (auto iattr = attr.dyn_cast<mlir::IntegerAttr>())
    return iattr.getInt() == conVal;
  return false;
}
static bool isZero(mlir::Attribute a) { return checkIsIntegerConstant(a, 0); }
static bool isOne(mlir::Attribute a) { return checkIsIntegerConstant(a, 1); }

// Undo some complex patterns created in the front-end and turn them back into
// complex ops.
template <typename FltOp, typename CpxOp>
struct UndoComplexPattern : public mlir::RewritePattern {
  UndoComplexPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern("fir.insert_value", 2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insval = dyn_cast_or_null<fir::InsertValueOp>(op);
    if (!insval || !insval.getType().isa<fir::ComplexType>())
      return mlir::failure();
    auto insval2 =
        dyn_cast_or_null<fir::InsertValueOp>(insval.adt().getDefiningOp());
    if (!insval2 || !isa<fir::UndefOp>(insval2.adt().getDefiningOp()))
      return mlir::failure();
    auto binf = dyn_cast_or_null<FltOp>(insval.val().getDefiningOp());
    auto binf2 = dyn_cast_or_null<FltOp>(insval2.val().getDefiningOp());
    if (!binf || !binf2 || insval.coor().size() != 1 ||
        !isOne(insval.coor()[0]) || insval2.coor().size() != 1 ||
        !isZero(insval2.coor()[0]))
      return mlir::failure();
    auto eai =
        dyn_cast_or_null<fir::ExtractValueOp>(binf.lhs().getDefiningOp());
    auto ebi =
        dyn_cast_or_null<fir::ExtractValueOp>(binf.rhs().getDefiningOp());
    auto ear =
        dyn_cast_or_null<fir::ExtractValueOp>(binf2.lhs().getDefiningOp());
    auto ebr =
        dyn_cast_or_null<fir::ExtractValueOp>(binf2.rhs().getDefiningOp());
    if (!eai || !ebi || !ear || !ebr || ear.adt() != eai.adt() ||
        ebr.adt() != ebi.adt() || eai.coor().size() != 1 ||
        !isOne(eai.coor()[0]) || ebi.coor().size() != 1 ||
        !isOne(ebi.coor()[0]) || ear.coor().size() != 1 ||
        !isZero(ear.coor()[0]) || ebr.coor().size() != 1 ||
        !isZero(ebr.coor()[0]))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<CpxOp>(op, ear.adt(), ebr.adt());
    return mlir::success();
  }
};

void fir::InsertValueOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<UndoComplexPattern<mlir::AddFOp, fir::AddcOp>,
                 UndoComplexPattern<mlir::SubFOp, fir::SubcOp>>(context);
}

//===----------------------------------------------------------------------===//
// IterWhileOp
//===----------------------------------------------------------------------===//

void fir::IterWhileOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value lb,
                             mlir::Value ub, mlir::Value step,
                             mlir::Value iterate, bool finalCountValue,
                             mlir::ValueRange iterArgs,
                             llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step, iterate});
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(getFinalValueAttrName(), builder.getUnitAttr());
  }
  result.addTypes(iterate.getType());
  result.addOperands(iterArgs);
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block{});
  bodyRegion->front().addArgument(builder.getIndexType());
  bodyRegion->front().addArgument(iterate.getType());
  bodyRegion->front().addArguments(iterArgs.getTypes());
  result.addAttributes(attributes);
}

static mlir::ParseResult parseIterWhileOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  if (parser.parseLParen() || parser.parseRegionArgument(inductionVariable) ||
      parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  auto i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.parseRParen() ||
      parser.resolveOperand(step, indexType, result.operands))
    return mlir::failure();

  mlir::OpAsmParser::OperandType iterateVar, iterateInput;
  if (parser.parseKeyword("and") || parser.parseLParen() ||
      parser.parseRegionArgument(iterateVar) || parser.parseEqual() ||
      parser.parseOperand(iterateInput) || parser.parseRParen() ||
      parser.resolveOperand(iterateInput, i1Type, result.operands))
    return mlir::failure();

  // Parse the initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::OperandType> regionArgs;
  auto prependCount = false;

  // Induction variable.
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(iterateVar);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    llvm::SmallVector<mlir::OpAsmParser::OperandType> operands;
    llvm::SmallVector<mlir::Type> regionTypes;
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(regionTypes))
      return failure();
    if (regionTypes.size() == operands.size() + 2)
      prependCount = true;
    llvm::ArrayRef<mlir::Type> resTypes = regionTypes;
    resTypes = prependCount ? resTypes.drop_front(2) : resTypes;
    // Resolve input operands.
    for (auto operandType : llvm::zip(operands, resTypes))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return failure();
    if (prependCount) {
      result.addTypes(regionTypes);
    } else {
      result.addTypes(i1Type);
      result.addTypes(resTypes);
    }
  } else if (succeeded(parser.parseOptionalArrow())) {
    llvm::SmallVector<mlir::Type> typeList;
    if (parser.parseLParen() || parser.parseTypeList(typeList) ||
        parser.parseRParen())
      return failure();
    // Type list must be "(index, i1)".
    if (typeList.size() != 2 || !typeList[0].isa<mlir::IndexType>() ||
        !typeList[1].isSignlessInteger(1))
      return failure();
    result.addTypes(typeList);
    prependCount = true;
  } else {
    result.addTypes(i1Type);
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes;
  // Induction variable (hidden)
  if (prependCount)
    result.addAttribute(IterWhileOp::getFinalValueAttrName(),
                        builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables (including iterate)
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  fir::IterWhileOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

static mlir::LogicalResult verify(fir::IterWhileOp op) {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(1).getType().isInteger(1))
    return op.emitOpError(
        "expected body second argument to be an index argument for "
        "the induction variable");
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (op.finalValue()) {
    // Result type must be "(index, i1, ...)".
    if (!op.getResult(0).getType().isa<mlir::IndexType>())
      return op.emitOpError("result #0 expected to be index");
    if (!op.getResult(1).getType().isSignlessInteger(1))
      return op.emitOpError("result #1 expected to be i1");
    opNumResults--;
  } else {
    // iterate_while always returns the early exit induction value.
    // Result type must be "(i1, ...)"
    if (!op.getResult(0).getType().isSignlessInteger(1))
      return op.emitOpError("result #0 expected to be i1");
  }
  if (opNumResults == 0)
    return mlir::failure();
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults =
      op.finalValue() ? op.getResults().drop_front() : op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::IterWhileOp op) {
  p << " (" << op.getInductionVar() << " = " << op.lowerBound() << " to "
    << op.upperBound() << " step " << op.step() << ") and (";
  assert(op.hasIterOperands());
  auto regionArgs = op.getRegionIterArgs();
  auto operands = op.getIterOperands();
  p << regionArgs.front() << " = " << *operands.begin() << ")";
  if (regionArgs.size() > 1) {
    p << " iter_args(";
    llvm::interleaveComma(
        llvm::zip(regionArgs.drop_front(), operands.drop_front()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ") -> (";
    llvm::interleaveComma(
        llvm::drop_begin(op.getResultTypes(), op.finalValue() ? 0 : 1), p);
    p << ")";
  } else if (op.finalValue()) {
    p << " -> (" << op.getResultTypes() << ')';
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {IterWhileOp::getFinalValueAttrName()});
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

mlir::Region &fir::IterWhileOp::getLoopBody() { return region(); }

bool fir::IterWhileOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

mlir::LogicalResult
fir::IterWhileOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

mlir::BlockArgument fir::IterWhileOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(initArgs()))
    if (iterArg == i.value())
      return region().front().getArgument(i.index() + 1);
  return {};
}

void fir::IterWhileOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = finalValue() ? resultNum + 1 : resultNum;
  auto *term = region().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

mlir::Value fir::IterWhileOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= initArgs().size())
    return initArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// LenParamIndexOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseLenParamIndexOp(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  mlir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return mlir::failure();
  result.addAttribute(fir::LenParamIndexOp::fieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!recty.dyn_cast<RecordType>())
    return mlir::failure();
  result.addAttribute(fir::LenParamIndexOp::typeAttrName(),
                      mlir::TypeAttr::get(recty));
  mlir::Type lenType = fir::LenType::get(builder.getContext());
  if (parser.addTypeToList(lenType, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::LenParamIndexOp &op) {
  p << ' '
    << op.getOperation()
           ->getAttrOfType<mlir::StringAttr>(
               fir::LenParamIndexOp::fieldAttrName())
           .getValue()
    << ", " << op.getOperation()->getAttr(fir::LenParamIndexOp::typeAttrName());
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void fir::LoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::Value refVal) {
  if (!refVal) {
    mlir::emitError(result.location, "LoadOp has null argument");
    return;
  }
  auto eleTy = fir::dyn_cast_ptrEleTy(refVal.getType());
  if (!eleTy) {
    mlir::emitError(result.location, "not a memory reference type");
    return;
  }
  result.addOperands(refVal);
  result.addTypes(eleTy);
}

/// Get the element type of a reference like type; otherwise null
static mlir::Type elementTypeOf(mlir::Type ref) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(ref)
      .Case<ReferenceType, PointerType, HeapType>(
          [](auto type) { return type.getEleTy(); })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

mlir::ParseResult fir::LoadOp::getElementOf(mlir::Type &ele, mlir::Type ref) {
  if ((ele = elementTypeOf(ref)))
    return mlir::success();
  return mlir::failure();
}

static mlir::ParseResult parseLoadOp(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::OperandType oper;
  if (parser.parseOperand(oper) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, type, result.operands))
    return mlir::failure();
  mlir::Type eleTy;
  if (fir::LoadOp::getElementOf(eleTy, type) ||
      parser.addTypeToList(eleTy, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::LoadOp &op) {
  p << ' ';
  p.printOperand(op.memref());
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), {});
  p << " : " << op.memref().getType();
}

//===----------------------------------------------------------------------===//
// DoLoopOp
//===----------------------------------------------------------------------===//

void fir::DoLoopOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Value lb,
                          mlir::Value ub, mlir::Value step, bool unordered,
                          bool finalCountValue, mlir::ValueRange iterArgs,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(finalValueAttrName(result.name), builder.getUnitAttr());
  }
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block{});
  if (iterArgs.empty() && !finalCountValue)
    DoLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  bodyRegion->front().addArgument(builder.getIndexType());
  bodyRegion->front().addArguments(iterArgs.getTypes());
  if (unordered)
    result.addAttribute(unorderedAttrName(result.name), builder.getUnitAttr());
  result.addAttributes(attributes);
}

static mlir::ParseResult parseDoLoopOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  if (mlir::succeeded(parser.parseOptionalKeyword("unordered")))
    result.addAttribute("unordered", builder.getUnitAttr());

  // Parse the optional initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::OperandType> regionArgs, operands;
  llvm::SmallVector<mlir::Type> argTypes;
  auto prependCount = false;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    if (result.types.size() == operands.size() + 1)
      prependCount = true;
    // Resolve input operands.
    llvm::ArrayRef<mlir::Type> resTypes = result.types;
    for (auto operand_type :
         llvm::zip(operands, prependCount ? resTypes.drop_front() : resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  } else if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseKeyword("index"))
      return failure();
    result.types.push_back(indexType);
    prependCount = true;
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  // Induction variable.
  if (prependCount)
    result.addAttribute(DoLoopOp::finalValueAttrName(result.name),
                        builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  DoLoopOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

fir::DoLoopOp fir::getForInductionVarOwner(mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg)
    return {};
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<fir::DoLoopOp>(containingInst);
}

// Lifted from loop.loop
static mlir::LogicalResult verify(fir::DoLoopOp op) {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();

  if (op.finalValue()) {
    if (op.unordered())
      return op.emitOpError("unordered loop has no final value");
    opNumResults--;
  }
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults =
      op.finalValue() ? op.getResults().drop_front() : op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }
  return success();
}

static void print(mlir::OpAsmPrinter &p, fir::DoLoopOp op) {
  bool printBlockTerminators = false;
  p << ' ' << op.getInductionVar() << " = " << op.lowerBound() << " to "
    << op.upperBound() << " step " << op.step();
  if (op.unordered())
    p << " unordered";
  if (op.hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = op.getRegionIterArgs();
    auto operands = op.getIterOperands();
    llvm::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ") -> (" << op.getResultTypes() << ')';
    printBlockTerminators = true;
  } else if (op.finalValue()) {
    p << " -> " << op.getResultTypes();
    printBlockTerminators = true;
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {"unordered", "finalValue"});
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
}

mlir::Region &fir::DoLoopOp::getLoopBody() { return region(); }

bool fir::DoLoopOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

mlir::LogicalResult
fir::DoLoopOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

/// Translate a value passed as an iter_arg to the corresponding block
/// argument in the body of the loop.
mlir::BlockArgument fir::DoLoopOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(initArgs()))
    if (iterArg == i.value())
      return region().front().getArgument(i.index() + 1);
  return {};
}

/// Translate the result vector (by index number) to the corresponding value
/// to the `fir.result` Op.
void fir::DoLoopOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = finalValue() ? resultNum + 1 : resultNum;
  auto *term = region().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

/// Translate the block argument (by index number) to the corresponding value
/// passed as an iter_arg to the parent DoLoopOp.
mlir::Value fir::DoLoopOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= initArgs().size())
    return initArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseDTEntryOp(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  llvm::StringRef methodName;
  // allow `methodName` or `"methodName"`
  if (failed(parser.parseOptionalKeyword(&methodName))) {
    mlir::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr, fir::DTEntryOp::getMethodAttrName(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::DTEntryOp::getMethodAttrName(),
                        parser.getBuilder().getStringAttr(methodName));
  }
  mlir::SymbolRefAttr calleeAttr;
  if (parser.parseComma() ||
      parser.parseAttribute(calleeAttr, fir::DTEntryOp::getProcAttrName(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::DTEntryOp &op) {
  p << ' ' << op.getOperation()->getAttr(fir::DTEntryOp::getMethodAttrName())
    << ", " << op.getOperation()->getAttr(fir::DTEntryOp::getProcAttrName());
}

//===----------------------------------------------------------------------===//
// ReboxOp
//===----------------------------------------------------------------------===//

/// Get the scalar type related to a fir.box type.
/// Example: return f32 for !fir.box<!fir.heap<!fir.array<?x?xf32>>.
static mlir::Type getBoxScalarEleTy(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getEleTy();
  return eleTy;
}

/// Get the rank from a !fir.box type
static unsigned getBoxRank(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getDimension();
  return 0;
}

static mlir::LogicalResult verify(fir::ReboxOp op) {
  auto inputBoxTy = op.box().getType();
  if (fir::isa_unknown_size_box(inputBoxTy))
    return op.emitOpError("box operand must not have unknown rank or type");
  auto outBoxTy = op.getType();
  if (fir::isa_unknown_size_box(outBoxTy))
    return op.emitOpError("result type must not have unknown rank or type");
  auto inputRank = getBoxRank(inputBoxTy);
  auto inputEleTy = getBoxScalarEleTy(inputBoxTy);
  auto outRank = getBoxRank(outBoxTy);
  auto outEleTy = getBoxScalarEleTy(outBoxTy);

  if (auto slice = op.slice()) {
    // Slicing case
    if (slice.getType().cast<fir::SliceType>().getRank() != inputRank)
      return op.emitOpError("slice operand rank must match box operand rank");
    if (auto shape = op.shape()) {
      if (auto shiftTy = shape.getType().dyn_cast<fir::ShiftType>()) {
        if (shiftTy.getRank() != inputRank)
          return op.emitOpError("shape operand and input box ranks must match "
                                "when there is a slice");
      } else {
        return op.emitOpError("shape operand must absent or be a fir.shift "
                              "when there is a slice");
      }
    }
    if (auto sliceOp = slice.getDefiningOp()) {
      auto slicedRank = mlir::cast<fir::SliceOp>(sliceOp).getOutRank();
      if (slicedRank != outRank)
        return op.emitOpError("result type rank and rank after applying slice "
                              "operand must match");
    }
  } else {
    // Reshaping case
    unsigned shapeRank = inputRank;
    if (auto shape = op.shape()) {
      auto ty = shape.getType();
      if (auto shapeTy = ty.dyn_cast<fir::ShapeType>()) {
        shapeRank = shapeTy.getRank();
      } else if (auto shapeShiftTy = ty.dyn_cast<fir::ShapeShiftType>()) {
        shapeRank = shapeShiftTy.getRank();
      } else {
        auto shiftTy = ty.cast<fir::ShiftType>();
        shapeRank = shiftTy.getRank();
        if (shapeRank != inputRank)
          return op.emitOpError("shape operand and input box ranks must match "
                                "when the shape is a fir.shift");
      }
    }
    if (shapeRank != outRank)
      return op.emitOpError("result type and shape operand ranks must match");
  }

  if (inputEleTy != outEleTy)
    // TODO: check that outBoxTy is a parent type of inputBoxTy for derived
    // types.
    if (!inputEleTy.isa<fir::RecordType>())
      return op.emitOpError(
          "op input and output element types must match for intrinsic types");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ResultOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ResultOp op) {
  auto *parentOp = op->getParentOp();
  auto results = parentOp->getResults();
  auto operands = op->getOperands();

  if (parentOp->getNumResults() != op.getNumOperands())
    return op.emitOpError() << "parent of result must have same arity";
  for (auto e : llvm::zip(results, operands))
    if (std::get<0>(e).getType() != std::get<1>(e).getType())
      return op.emitOpError()
             << "types mismatch between result op and its parent";
  return success();
}

//===----------------------------------------------------------------------===//
// SaveResultOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::SaveResultOp op) {
  auto resultType = op.value().getType();
  if (resultType != fir::dyn_cast_ptrEleTy(op.memref().getType()))
    return op.emitOpError("value type must match memory reference type");
  if (fir::isa_unknown_size_box(resultType))
    return op.emitOpError("cannot save !fir.box of unknown rank or type");

  if (resultType.isa<fir::BoxType>()) {
    if (op.shape() || !op.typeparams().empty())
      return op.emitOpError(
          "must not have shape or length operands if the value is a fir.box");
    return mlir::success();
  }

  // fir.record or fir.array case.
  unsigned shapeTyRank = 0;
  if (auto shapeOp = op.shape()) {
    auto shapeTy = shapeOp.getType();
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>())
      shapeTyRank = s.getRank();
    else
      shapeTyRank = shapeTy.cast<fir::ShapeShiftType>().getRank();
  }

  auto eleTy = resultType;
  if (auto seqTy = resultType.dyn_cast<fir::SequenceType>()) {
    if (seqTy.getDimension() != shapeTyRank)
      op.emitOpError("shape operand must be provided and have the value rank "
                     "when the value is a fir.array");
    eleTy = seqTy.getEleTy();
  } else {
    if (shapeTyRank != 0)
      op.emitOpError(
          "shape operand should only be provided if the value is a fir.array");
  }

  if (auto recTy = eleTy.dyn_cast<fir::RecordType>()) {
    if (recTy.getNumLenParams() != op.typeparams().size())
      op.emitOpError("length parameters number must match with the value type "
                     "length parameters");
  } else if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
    if (op.typeparams().size() > 1)
      op.emitOpError("no more than one length parameter must be provided for "
                     "character value");
  } else {
    if (!op.typeparams().empty())
      op.emitOpError(
          "length parameters must not be provided for this value type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

static constexpr llvm::StringRef getCompareOffsetAttr() {
  return "compare_operand_offsets";
}

static constexpr llvm::StringRef getTargetOffsetAttr() {
  return "target_operand_offsets";
}

template <typename A, typename... AdditionalArgs>
static A getSubOperands(unsigned pos, A allArgs,
                        mlir::DenseIntElementsAttr ranges,
                        AdditionalArgs &&...additionalArgs) {
  unsigned start = 0;
  for (unsigned i = 0; i < pos; ++i)
    start += (*(ranges.begin() + i)).getZExtValue();
  return allArgs.slice(start, (*(ranges.begin() + pos)).getZExtValue(),
                       std::forward<AdditionalArgs>(additionalArgs)...);
}

static mlir::MutableOperandRange
getMutableSuccessorOperands(unsigned pos, mlir::MutableOperandRange operands,
                            StringRef offsetAttr) {
  Operation *owner = operands.getOwner();
  NamedAttribute targetOffsetAttr =
      *owner->getAttrDictionary().getNamed(offsetAttr);
  return getSubOperands(
      pos, operands, targetOffsetAttr.second.cast<DenseIntElementsAttr>(),
      mlir::MutableOperandRange::OperandSegment(pos, targetOffsetAttr));
}

static unsigned denseElementsSize(mlir::DenseIntElementsAttr attr) {
  return attr.getNumElements();
}

llvm::Optional<mlir::OperandRange> fir::SelectOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                    unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectCaseOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectCaseOp::getCompareOperands(unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  return {getSubOperands(cond, compareArgs(), a)};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getCompareOperands(llvm::ArrayRef<mlir::Value> operands,
                                      unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectCaseOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

// parser for fir.select_case Op
static mlir::ParseResult parseSelectCase(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType selector;
  mlir::Type type;
  if (parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::SmallVector<mlir::OpAsmParser::OperandType> opers;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  llvm::SmallVector<int32_t> argOffs;
  int32_t offSize = 0;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || isValidCaseAttr(attr) ||
        parser.parseComma())
      return mlir::failure();
    attrs.push_back(attr);
    if (attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      argOffs.push_back(0);
    } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
      mlir::OpAsmParser::OperandType oper1;
      mlir::OpAsmParser::OperandType oper2;
      if (parser.parseOperand(oper1) || parser.parseComma() ||
          parser.parseOperand(oper2) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper1);
      opers.push_back(oper2);
      argOffs.push_back(2);
      offSize += 2;
    } else {
      mlir::OpAsmParser::OperandType oper;
      if (parser.parseOperand(oper) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper);
      argOffs.push_back(1);
      ++offSize;
    }
    if (parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  result.addAttribute(fir::SelectCaseOp::getCasesAttr(),
                      parser.getBuilder().getArrayAttr(attrs));
  if (parser.resolveOperands(opers, type, result.operands))
    return mlir::failure();
  llvm::SmallVector<int32_t> targOffs;
  int32_t toffSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    targOffs.push_back(argSize);
    toffSize += argSize;
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectCaseOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, offSize, toffSize}));
  result.addAttribute(getCompareOffsetAttr(), bld.getI32VectorAttr(argOffs));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(targOffs));
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::SelectCaseOp &op) {
  p << ' ';
  p.printOperand(op.getSelector());
  p << " : " << op.getSelector().getType() << " [";
  auto cases = op.getOperation()
                   ->getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr())
                   .getValue();
  auto count = op.getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    if (!cases[i].isa<mlir::UnitAttr>()) {
      auto caseArgs = *op.getCompareOperands(i);
      p.printOperand(*caseArgs.begin());
      p << ", ";
      if (cases[i].isa<fir::ClosedIntervalAttr>()) {
        p.printOperand(*(++caseArgs.begin()));
        p << ", ";
      }
    }
    op.printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(op.getOperation()->getAttrs(),
                          {op.getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(),
                           op.getOperandSegmentSizeAttr()});
}

unsigned fir::SelectCaseOp::compareOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr()));
}

unsigned fir::SelectCaseOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::ValueRange> cmpOperands,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(compareAttrs));
  llvm::SmallVector<int32_t> operOffs;
  int32_t operSize = 0;
  for (auto attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      operOffs.push_back(2);
      operSize += 2;
    } else if (attr.isa<mlir::UnitAttr>()) {
      operOffs.push_back(0);
    } else {
      operOffs.push_back(1);
      ++operSize;
    }
  }
  for (auto ops : cmpOperands)
    result.addOperands(ops);
  result.addAttribute(getCompareOffsetAttr(),
                      builder.getI32VectorAttr(operOffs));
  const auto count = destinations.size();
  for (auto d : destinations)
    result.addSuccessors(d);
  const auto opCount = destOperands.size();
  llvm::SmallVector<int32_t> argOffs;
  int32_t sumArgs = 0;
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    if (i < opCount) {
      result.addOperands(destOperands[i]);
      const auto argSz = destOperands[i].size();
      argOffs.push_back(argSz);
      sumArgs += argSz;
    } else {
      argOffs.push_back(0);
    }
  }
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({1, operSize, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), builder.getI32VectorAttr(argOffs));
  result.addAttributes(attributes);
}

/// This builder has a slightly simplified interface in that the list of
/// operands need not be partitioned by the builder. Instead the operands are
/// partitioned here, before being passed to the default builder. This
/// partitioning is unchecked, so can go awry on bad input.
void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::Value> cmpOpList,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  llvm::SmallVector<mlir::ValueRange> cmpOpers;
  auto iter = cmpOpList.begin();
  for (auto &attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 2}));
      iter += 2;
    } else if (attr.isa<UnitAttr>()) {
      cmpOpers.push_back(mlir::ValueRange{});
    } else {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 1}));
      ++iter;
    }
  }
  build(builder, result, selector, compareAttrs, cmpOpers, destinations,
        destOperands, attributes);
}

static mlir::LogicalResult verify(fir::SelectCaseOp &op) {
  if (!(op.getSelector().getType().isa<mlir::IntegerType>() ||
        op.getSelector().getType().isa<mlir::IndexType>() ||
        op.getSelector().getType().isa<fir::IntegerType>() ||
        op.getSelector().getType().isa<fir::LogicalType>() ||
        op.getSelector().getType().isa<fir::CharacterType>()))
    return op.emitOpError("must be an integer, character, or logical");
  auto cases = op.getOperation()
                   ->getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr())
                   .getValue();
  auto count = op.getNumDest();
  if (count == 0)
    return op.emitOpError("must have at least one successor");
  if (op.getNumConditions() != count)
    return op.emitOpError("number of conditions and successors don't match");
  if (op.compareOffsetSize() != count)
    return op.emitOpError("incorrect number of compare operand groups");
  if (op.targetOffsetSize() != count)
    return op.emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    auto &attr = cases[i];
    if (!(attr.isa<fir::PointIntervalAttr>() ||
          attr.isa<fir::LowerBoundAttr>() || attr.isa<fir::UpperBoundAttr>() ||
          attr.isa<fir::ClosedIntervalAttr>() || attr.isa<mlir::UnitAttr>()))
      return op.emitOpError("incorrect select case attribute type");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SelectRankOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectRankOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectRankOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectRankOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectTypeOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectTypeOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectTypeOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

static ParseResult parseSelectType(OpAsmParser &parser,
                                   OperationState &result) {
  mlir::OpAsmParser::OperandType selector;
  mlir::Type type;
  if (parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    attrs.push_back(attr);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectTypeOp::getCasesAttr(),
                      bld.getArrayAttr(attrs));
  llvm::SmallVector<int32_t> argOffs;
  int32_t offSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    argOffs.push_back(argSize);
    offSize += argSize;
  }
  result.addAttribute(fir::SelectTypeOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, 0, offSize}));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(argOffs));
  return mlir::success();
}

unsigned fir::SelectTypeOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

static void print(mlir::OpAsmPrinter &p, fir::SelectTypeOp &op) {
  p << ' ';
  p.printOperand(op.getSelector());
  p << " : " << op.getSelector().getType() << " [";
  auto cases = op.getOperation()
                   ->getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr())
                   .getValue();
  auto count = op.getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    op.printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(op.getOperation()->getAttrs(),
                          {op.getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(),
                           fir::SelectTypeOp::getOperandSegmentSizeAttr()});
}

static mlir::LogicalResult verify(fir::SelectTypeOp &op) {
  if (!(op.getSelector().getType().isa<fir::BoxType>()))
    return op.emitOpError("must be a boxed type");
  auto cases = op.getOperation()
                   ->getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr())
                   .getValue();
  auto count = op.getNumDest();
  if (count == 0)
    return op.emitOpError("must have at least one successor");
  if (op.getNumConditions() != count)
    return op.emitOpError("number of conditions and successors don't match");
  if (op.targetOffsetSize() != count)
    return op.emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    auto &attr = cases[i];
    if (!(attr.isa<fir::ExactTypeAttr>() || attr.isa<fir::SubclassAttr>() ||
          attr.isa<mlir::UnitAttr>()))
      return op.emitOpError("invalid type-case alternative");
  }
  return mlir::success();
}

void fir::SelectTypeOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> typeOperands,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(typeOperands));
  const auto count = destinations.size();
  for (mlir::Block *dest : destinations)
    result.addSuccessors(dest);
  const auto opCount = destOperands.size();
  llvm::SmallVector<int32_t> argOffs;
  int32_t sumArgs = 0;
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    if (i < opCount) {
      result.addOperands(destOperands[i]);
      const auto argSz = destOperands[i].size();
      argOffs.push_back(argSz);
      sumArgs += argSz;
    } else {
      argOffs.push_back(0);
    }
  }
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({1, 0, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), builder.getI32VectorAttr(argOffs));
  result.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ShapeOp &op) {
  auto size = op.extents().size();
  auto shapeTy = op.getType().dyn_cast<fir::ShapeType>();
  assert(shapeTy && "must be a shape type");
  if (shapeTy.getRank() != size)
    return op.emitOpError("shape type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShapeShiftOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ShapeShiftOp &op) {
  auto size = op.pairs().size();
  if (size < 2 || size > 16 * 2)
    return op.emitOpError("incorrect number of args");
  if (size % 2 != 0)
    return op.emitOpError("requires a multiple of 2 args");
  auto shapeTy = op.getType().dyn_cast<fir::ShapeShiftType>();
  assert(shapeTy && "must be a shape shift type");
  if (shapeTy.getRank() * 2 != size)
    return op.emitOpError("shape type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ShiftOp &op) {
  auto size = op.origins().size();
  auto shiftTy = op.getType().dyn_cast<fir::ShiftType>();
  assert(shiftTy && "must be a shift type");
  if (shiftTy.getRank() != size)
    return op.emitOpError("shift type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

/// Return the output rank of a slice op. The output rank must be between 1 and
/// the rank of the array being sliced (inclusive).
unsigned fir::SliceOp::getOutputRank(mlir::ValueRange triples) {
  unsigned rank = 0;
  if (!triples.empty()) {
    for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
      auto op = triples[i].getDefiningOp();
      if (!mlir::isa_and_nonnull<fir::UndefOp>(op))
        ++rank;
    }
    assert(rank > 0);
  }
  return rank;
}

static mlir::LogicalResult verify(fir::SliceOp &op) {
  auto size = op.triples().size();
  if (size < 3 || size > 16 * 3)
    return op.emitOpError("incorrect number of args for triple");
  if (size % 3 != 0)
    return op.emitOpError("requires a multiple of 3 args");
  auto sliceTy = op.getType().dyn_cast<fir::SliceType>();
  assert(sliceTy && "must be a slice type");
  if (sliceTy.getRank() * 3 != size)
    return op.emitOpError("slice type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

mlir::Type fir::StoreOp::elementType(mlir::Type refType) {
  return fir::dyn_cast_ptrEleTy(refType);
}

static mlir::ParseResult parseStoreOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::OperandType oper;
  mlir::OpAsmParser::OperandType store;
  if (parser.parseOperand(oper) || parser.parseKeyword("to") ||
      parser.parseOperand(store) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, fir::StoreOp::elementType(type),
                            result.operands) ||
      parser.resolveOperand(store, type, result.operands))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::StoreOp &op) {
  p << ' ';
  p.printOperand(op.value());
  p << " to ";
  p.printOperand(op.memref());
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), {});
  p << " : " << op.memref().getType();
}

static mlir::LogicalResult verify(fir::StoreOp &op) {
  if (op.value().getType() != fir::dyn_cast_ptrEleTy(op.memref().getType()))
    return op.emitOpError("store value type must match memory reference type");
  if (fir::isa_unknown_size_box(op.value().getType()))
    return op.emitOpError("cannot store !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StringLitOp
//===----------------------------------------------------------------------===//

bool fir::StringLitOp::isWideValue() {
  auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
  return eleTy.cast<fir::CharacterType>().getFKind() != 1;
}

static mlir::NamedAttribute
mkNamedIntegerAttr(mlir::OpBuilder &builder, llvm::StringRef name, int64_t v) {
  assert(v > 0);
  return builder.getNamedAttr(
      name, builder.getIntegerAttr(builder.getIntegerType(64), v));
}

void fir::StringLitOp::build(mlir::OpBuilder &builder, OperationState &result,
                             fir::CharacterType inType, llvm::StringRef val,
                             llvm::Optional<int64_t> len) {
  auto valAttr = builder.getNamedAttr(value(), builder.getStringAttr(val));
  int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

template <typename C>
static mlir::ArrayAttr convertToArrayAttr(mlir::OpBuilder &builder,
                                          llvm::ArrayRef<C> xlist) {
  llvm::SmallVector<mlir::Attribute> attrs;
  auto ty = builder.getIntegerType(8 * sizeof(C));
  for (auto ch : xlist)
    attrs.push_back(builder.getIntegerAttr(ty, ch));
  return builder.getArrayAttr(attrs);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder, OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char> vlist,
                             llvm::Optional<int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder, OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char16_t> vlist,
                             llvm::Optional<int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder, OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char32_t> vlist,
                             llvm::Optional<int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len.hasValue() ? len.getValue() : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

static mlir::ParseResult parseStringLitOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::Attribute val;
  mlir::NamedAttrList attrs;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseAttribute(val, "fake", attrs))
    return mlir::failure();
  if (auto v = val.dyn_cast<mlir::StringAttr>())
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::value(), v));
  else if (auto v = val.dyn_cast<mlir::ArrayAttr>())
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::xlist(), v));
  else
    return parser.emitError(parser.getCurrentLocation(),
                            "found an invalid constant");
  mlir::IntegerAttr sz;
  mlir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(sz, fir::StringLitOp::size(), result.attributes) ||
      parser.parseRParen() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseColonType(type))
    return mlir::failure();
  auto charTy = type.dyn_cast<fir::CharacterType>();
  if (!charTy)
    return parser.emitError(trailingTypeLoc, "must have character type");
  type = fir::CharacterType::get(builder.getContext(), charTy.getFKind(),
                                 sz.getInt());
  if (!type || parser.addTypesToList(type, result.types))
    return mlir::failure();
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::StringLitOp &op) {
  p << ' ' << op.getValue() << '(';
  p << op.getSize().cast<mlir::IntegerAttr>().getValue() << ") : ";
  p.printType(op.getType());
}

static mlir::LogicalResult verify(fir::StringLitOp &op) {
  if (op.getSize().cast<mlir::IntegerAttr>().getValue().isNegative())
    return op.emitOpError("size must be non-negative");
  if (auto xl = op.getOperation()->getAttr(fir::StringLitOp::xlist())) {
    auto xList = xl.cast<mlir::ArrayAttr>();
    for (auto a : xList)
      if (!a.isa<mlir::IntegerAttr>())
        return op.emitOpError("values in list must be integers");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnboxProcOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::UnboxProcOp &op) {
  if (auto eleTy = fir::dyn_cast_ptrEleTy(op.refTuple().getType()))
    if (eleTy.isa<mlir::TupleType>())
      return mlir::success();
  return op.emitOpError("second output argument has bad type");
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void fir::IfOp::build(mlir::OpBuilder &builder, OperationState &result,
                      mlir::Value cond, bool withElseRegion) {
  build(builder, result, llvm::None, cond, withElseRegion);
}

void fir::IfOp::build(mlir::OpBuilder &builder, OperationState &result,
                      mlir::TypeRange resultTypes, mlir::Value cond,
                      bool withElseRegion) {
  result.addOperands(cond);
  result.addTypes(resultTypes);

  mlir::Region *thenRegion = result.addRegion();
  thenRegion->push_back(new mlir::Block());
  if (resultTypes.empty())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  mlir::Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new mlir::Block());
    if (resultTypes.empty())
      IfOp::ensureTerminator(*elseRegion, builder, result.location);
  }
}

static mlir::ParseResult parseIfOp(OpAsmParser &parser,
                                   OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return mlir::failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return mlir::failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  return mlir::success();
}

static LogicalResult verify(fir::IfOp op) {
  if (op.getNumResults() != 0 && op.elseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::IfOp op) {
  bool printBlockTerminators = false;
  p << ' ' << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ')';
    printBlockTerminators = true;
  }
  p.printRegion(op.thenRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &otherReg = op.elseRegion();
  if (!otherReg.empty()) {
    p << " else";
    p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
                  printBlockTerminators);
  }
  p.printOptionalAttrDict(op->getAttrs());
}

void fir::IfOp::resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                                  unsigned resultNum) {
  auto *term = thenRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
  term = elseRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
}

//===----------------------------------------------------------------------===//

mlir::ParseResult fir::isValidCaseAttr(mlir::Attribute attr) {
  if (attr.dyn_cast_or_null<mlir::UnitAttr>() ||
      attr.dyn_cast_or_null<ClosedIntervalAttr>() ||
      attr.dyn_cast_or_null<PointIntervalAttr>() ||
      attr.dyn_cast_or_null<LowerBoundAttr>() ||
      attr.dyn_cast_or_null<UpperBoundAttr>())
    return mlir::success();
  return mlir::failure();
}

unsigned fir::getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                                    unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<ClosedIntervalAttr>())
        ++o;
    }
  }
  return o;
}

mlir::ParseResult fir::parseSelector(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result,
                                     mlir::OpAsmParser::OperandType &selector,
                                     mlir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return mlir::failure();
  return mlir::success();
}

/// Generic pretty-printer of a binary operation
static void printBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op must have two operands");
  assert(op->getNumResults() == 1 && "binary op must have one result");

  p << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

/// Generic pretty-printer of an unary operation
static void printUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op must have one operand");
  assert(op->getNumResults() == 1 && "unary op must have one result");

  p << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

bool fir::isReferenceLike(mlir::Type type) {
  return type.isa<fir::ReferenceType>() || type.isa<fir::HeapType>() ||
         type.isa<fir::PointerType>();
}

mlir::FuncOp fir::createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                               StringRef name, mlir::FunctionType type,
                               llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto f = module.lookupSymbol<mlir::FuncOp>(name))
    return f;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPoint(module.getBody()->getTerminator());
  auto result = modBuilder.create<mlir::FuncOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

fir::GlobalOp fir::createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                                  StringRef name, mlir::Type type,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto g = module.lookupSymbol<fir::GlobalOp>(name))
    return g;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  auto result = modBuilder.create<fir::GlobalOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

bool fir::valueHasFirAttribute(mlir::Value value,
                               llvm::StringRef attributeName) {
  // If this is a fir.box that was loaded, the fir attributes will be on the
  // related fir.ref<fir.box> creation.
  if (value.getType().isa<fir::BoxType>())
    if (auto definingOp = value.getDefiningOp())
      if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(definingOp))
        value = loadOp.memref();
  // If this is a function argument, look in the argument attributes.
  if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock())
      if (auto funcOp =
              mlir::dyn_cast<mlir::FuncOp>(blockArg.getOwner()->getParentOp()))
        if (funcOp.getArgAttr(blockArg.getArgNumber(), attributeName))
          return true;
    return false;
  }

  if (auto definingOp = value.getDefiningOp()) {
    // If this is an allocated value, look at the allocation attributes.
    if (mlir::isa<fir::AllocMemOp>(definingOp) ||
        mlir::isa<AllocaOp>(definingOp))
      return definingOp->hasAttr(attributeName);
    // If this is an imported global, look at AddrOfOp and GlobalOp attributes.
    // Both operations are looked at because use/host associated variable (the
    // AddrOfOp) can have ASYNCHRONOUS/VOLATILE attributes even if the ultimate
    // entity (the globalOp) does not have them.
    if (auto addressOfOp = mlir::dyn_cast<fir::AddrOfOp>(definingOp)) {
      if (addressOfOp->hasAttr(attributeName))
        return true;
      if (auto module = definingOp->getParentOfType<mlir::ModuleOp>())
        if (auto globalOp =
                module.lookupSymbol<fir::GlobalOp>(addressOfOp.symbol()))
          return globalOp->hasAttr(attributeName);
    }
  }
  // TODO: Construct associated entities attributes. Decide where the fir
  // attributes must be placed/looked for in this case.
  return false;
}

mlir::Type fir::applyPathToType(mlir::Type eleTy, mlir::ValueRange path) {
  for (auto i = path.begin(), end = path.end(); eleTy && i < end;) {
    eleTy = llvm::TypeSwitch<mlir::Type, mlir::Type>(eleTy)
                .Case<fir::RecordType>([&](fir::RecordType ty) {
                  if (auto *op = (*i++).getDefiningOp()) {
                    if (auto off = mlir::dyn_cast<fir::FieldIndexOp>(op))
                      return ty.getType(off.getFieldName());
                    if (auto off = mlir::dyn_cast<mlir::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  }
                  return mlir::Type{};
                })
                .Case<fir::SequenceType>([&](fir::SequenceType ty) {
                  bool valid = true;
                  const auto rank = ty.getDimension();
                  for (std::remove_const_t<decltype(rank)> ii = 0;
                       valid && ii < rank; ++ii)
                    valid = i < end && fir::isa_integer((*i++).getType());
                  return valid ? ty.getEleTy() : mlir::Type{};
                })
                .Case<mlir::TupleType>([&](mlir::TupleType ty) {
                  if (auto *op = (*i++).getDefiningOp())
                    if (auto off = mlir::dyn_cast<mlir::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  return mlir::Type{};
                })
                .Case<fir::ComplexType>([&](fir::ComplexType ty) {
                  if (fir::isa_integer((*i++).getType()))
                    return ty.getElementType();
                  return mlir::Type{};
                })
                .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
                  if (fir::isa_integer((*i++).getType()))
                    return ty.getElementType();
                  return mlir::Type{};
                })
                .Default([&](const auto &) { return mlir::Type{}; });
  }
  return eleTy;
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
