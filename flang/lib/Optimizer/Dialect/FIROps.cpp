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
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

namespace {
#include "flang/Optimizer/Dialect/CanonicalizationPatterns.inc"
} // namespace

static llvm::cl::opt<bool> clUseStrictVolatileVerification(
    "strict-fir-volatile-verifier", llvm::cl::init(false),
    llvm::cl::desc(
        "use stricter verifier for FIR operations with volatile types"));

bool fir::useStrictVolatileVerification() {
  return clUseStrictVolatileVerification;
}

static void propagateAttributes(mlir::Operation *fromOp,
                                mlir::Operation *toOp) {
  if (!fromOp || !toOp)
    return;

  for (mlir::NamedAttribute attr : fromOp->getAttrs()) {
    if (attr.getName().getValue().starts_with(
            mlir::acc::OpenACCDialect::getDialectNamespace()))
      toOp->setAttr(attr.getName(), attr.getValue());
  }
}

/// Return true if a sequence type is of some incomplete size or a record type
/// is malformed or contains an incomplete sequence type. An incomplete sequence
/// type is one with more unknown extents in the type than have been provided
/// via `dynamicExtents`. Sequence types with an unknown rank are incomplete by
/// definition.
static bool verifyInType(mlir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited,
                         unsigned dynamicExtents = 0) {
  if (auto st = mlir::dyn_cast<fir::SequenceType>(inType)) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (std::size_t i = 0, end = shape.size(); i < end; ++i) {
      if (shape[i] != fir::SequenceType::getUnknownExtent())
        continue;
      if (dynamicExtents-- == 0)
        return true;
    }
  } else if (auto rt = mlir::dyn_cast<fir::RecordType>(inType)) {
    // don't recurse if we're already visiting this one
    if (llvm::is_contained(visited, rt.getName()))
      return false;
    // keep track of record types currently being visited
    visited.push_back(rt.getName());
    for (auto &field : rt.getTypeList())
      if (verifyInType(field.second, visited))
        return true;
    visited.pop_back();
  }
  return false;
}

static bool verifyTypeParamCount(mlir::Type inType, unsigned numParams) {
  auto ty = fir::unwrapSequenceType(inType);
  if (numParams > 0) {
    if (auto recTy = mlir::dyn_cast<fir::RecordType>(ty))
      return numParams != recTy.getNumLenParams();
    if (auto chrTy = mlir::dyn_cast<fir::CharacterType>(ty))
      return !(numParams == 1 && chrTy.hasDynamicLen());
    return true;
  }
  if (auto chrTy = mlir::dyn_cast<fir::CharacterType>(ty))
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
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
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
  result.addAttribute("operandSegmentSizes", builder.getDenseI32ArrayAttr(
                                                 {typeparamsSize, shapeSize}));
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

template <typename OP>
static void printAllocatableOp(mlir::OpAsmPrinter &p, OP &op) {
  p << ' ' << op.getInType();
  if (!op.getTypeparams().empty()) {
    p << '(' << op.getTypeparams() << " : " << op.getTypeparams().getTypes()
      << ')';
  }
  // print the shape of the allocation (if any); all must be index type
  for (auto sh : op.getShape()) {
    p << ", ";
    p.printOperand(sh);
  }
  p.printOptionalAttrDict(op->getAttrs(), {"in_type", "operandSegmentSizes"});
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Create a legal memory reference as return type
static mlir::Type wrapAllocaResultType(mlir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (mlir::isa<fir::ReferenceType>(intype))
    return {};
  return fir::ReferenceType::get(intype);
}

mlir::Type fir::AllocaOp::getAllocatedType() {
  return mlir::cast<fir::ReferenceType>(getType()).getEleTy();
}

mlir::Type fir::AllocaOp::getRefTy(mlir::Type ty) {
  return fir::ReferenceType::get(ty);
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

mlir::ParseResult fir::AllocaOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseAllocatableOp(wrapAllocaResultType, parser, result);
}

void fir::AllocaOp::print(mlir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

llvm::LogicalResult fir::AllocaOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  mlir::Type outType = getType();
  if (!mlir::isa<fir::ReferenceType>(outType))
    return emitOpError("must be a !fir.ref type");
  return mlir::success();
}

bool fir::AllocaOp::ownsNestedAlloca(mlir::Operation *op) {
  return op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>() ||
         op->hasTrait<mlir::OpTrait::AutomaticAllocationScope>() ||
         mlir::isa<mlir::LoopLikeOpInterface>(*op);
}

mlir::Region *fir::AllocaOp::getOwnerRegion() {
  mlir::Operation *currentOp = getOperation();
  while (mlir::Operation *parentOp = currentOp->getParentOp()) {
    // If the operation was not registered, inquiries about its traits will be
    // incorrect and it is not possible to reason about the operation. This
    // should not happen in a normal Fortran compilation flow, but be foolproof.
    if (!parentOp->isRegistered())
      return nullptr;
    if (fir::AllocaOp::ownsNestedAlloca(parentOp))
      return currentOp->getParentRegion();
    currentOp = parentOp;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// AllocMemOp
//===----------------------------------------------------------------------===//

/// Create a legal heap reference as return type
static mlir::Type wrapAllocMemResultType(mlir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (mlir::isa<fir::ReferenceType, fir::HeapType, fir::PointerType,
                mlir::FunctionType>(intype))
    return {};
  return fir::HeapType::get(intype);
}

mlir::Type fir::AllocMemOp::getAllocatedType() {
  return mlir::cast<fir::HeapType>(getType()).getEleTy();
}

mlir::Type fir::AllocMemOp::getRefTy(mlir::Type ty) {
  return fir::HeapType::get(ty);
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

mlir::ParseResult fir::AllocMemOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  return parseAllocatableOp(wrapAllocMemResultType, parser, result);
}

void fir::AllocMemOp::print(mlir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

llvm::LogicalResult fir::AllocMemOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  mlir::Type outType = getType();
  if (!mlir::dyn_cast<fir::HeapType>(outType))
    return emitOpError("must be a !fir.heap type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return emitOpError("cannot allocate !fir.box of unknown rank or type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayCoorOp
//===----------------------------------------------------------------------===//

// CHARACTERs and derived types with LEN PARAMETERs are dependent types that
// require runtime values to fully define the type of an object.
static bool validTypeParams(mlir::Type dynTy, mlir::ValueRange typeParams,
                            bool allowParamsForBox = false) {
  dynTy = fir::unwrapAllRefAndSeqType(dynTy);
  if (mlir::isa<fir::BaseBoxType>(dynTy)) {
    // A box value will contain type parameter values itself.
    if (!allowParamsForBox)
      return typeParams.size() == 0;

    // A boxed value may have no length parameters, when the lengths
    // are assumed. If dynamic lengths are used, then proceed
    // to the verification below.
    if (typeParams.size() == 0)
      return true;

    dynTy = fir::getFortranElementType(dynTy);
  }
  // Derived type must have all type parameters satisfied.
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(dynTy))
    return typeParams.size() == recTy.getNumLenParams();
  // Characters with non-constant LEN must have a type parameter value.
  if (auto charTy = mlir::dyn_cast<fir::CharacterType>(dynTy))
    if (charTy.hasDynamicLen())
      return typeParams.size() == 1;
  // Otherwise, any type parameters are invalid.
  return typeParams.size() == 0;
}

llvm::LogicalResult fir::ArrayCoorOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = mlir::dyn_cast<fir::SequenceType>(eleTy);
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = mlir::dyn_cast<fir::ShapeType>(shapeTy)) {
      shapeTyRank = s.getRank();
    } else if (auto ss = mlir::dyn_cast<fir::ShapeShiftType>(shapeTy)) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = mlir::cast<fir::ShiftType>(shapeTy);
      shapeTyRank = s.getRank();
      // TODO: it looks like PreCGRewrite and CodeGen can support
      // fir.shift with plain array reference, so we may consider
      // removing this check.
      if (!mlir::isa<fir::BaseBoxType>(getMemref().getType()))
        return emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return emitOpError("rank of dimension mismatched");
    // TODO: support slicing with changing the number of dimensions,
    // e.g. when array_coor represents an element access to array(:,1,:)
    // slice: the shape is 3D and the number of indices is 2 in this case.
    if (shapeTyRank != getIndices().size())
      return emitOpError("number of indices do not match dim rank");
  }

  if (auto sliceOp = getSlice()) {
    if (auto sl = mlir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_coor cannot take a slice with substring");
    if (auto sliceTy = mlir::dyn_cast<fir::SliceType>(sliceOp.getType()))
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }
  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");

  return mlir::success();
}

// Pull in fir.embox and fir.rebox into fir.array_coor when possible.
struct SimplifyArrayCoorOp : public mlir::OpRewritePattern<fir::ArrayCoorOp> {
  using mlir::OpRewritePattern<fir::ArrayCoorOp>::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(fir::ArrayCoorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value memref = op.getMemref();
    if (!mlir::isa<fir::BaseBoxType>(memref.getType()))
      return mlir::failure();

    mlir::Value boxedMemref, boxedShape, boxedSlice;
    if (auto emboxOp =
            mlir::dyn_cast_or_null<fir::EmboxOp>(memref.getDefiningOp())) {
      boxedMemref = emboxOp.getMemref();
      boxedShape = emboxOp.getShape();
      boxedSlice = emboxOp.getSlice();
      // If any of operands, that are not currently supported for migration
      // to ArrayCoorOp, is present, don't rewrite.
      if (!emboxOp.getTypeparams().empty() || emboxOp.getSourceBox() ||
          emboxOp.getAccessMap())
        return mlir::failure();
    } else if (auto reboxOp = mlir::dyn_cast_or_null<fir::ReboxOp>(
                   memref.getDefiningOp())) {
      boxedMemref = reboxOp.getBox();
      boxedShape = reboxOp.getShape();
      // Avoid pulling in rebox that performs reshaping.
      // There is no way to represent box reshaping with array_coor.
      if (boxedShape && !mlir::isa<fir::ShiftType>(boxedShape.getType()))
        return mlir::failure();
      boxedSlice = reboxOp.getSlice();
    } else {
      return mlir::failure();
    }

    bool boxedShapeIsShift =
        boxedShape && mlir::isa<fir::ShiftType>(boxedShape.getType());
    bool boxedShapeIsShape =
        boxedShape && mlir::isa<fir::ShapeType>(boxedShape.getType());
    bool boxedShapeIsShapeShift =
        boxedShape && mlir::isa<fir::ShapeShiftType>(boxedShape.getType());

    // Slices changing the number of dimensions are not supported
    // for array_coor yet.
    unsigned origBoxRank;
    if (mlir::isa<fir::BaseBoxType>(boxedMemref.getType()))
      origBoxRank = fir::getBoxRank(boxedMemref.getType());
    else if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(
                 fir::unwrapRefType(boxedMemref.getType())))
      origBoxRank = arrTy.getDimension();
    else
      return mlir::failure();

    if (fir::getBoxRank(memref.getType()) != origBoxRank)
      return mlir::failure();

    // Slices with substring are not supported by array_coor.
    if (boxedSlice)
      if (auto sliceOp =
              mlir::dyn_cast_or_null<fir::SliceOp>(boxedSlice.getDefiningOp()))
        if (!sliceOp.getSubstr().empty())
          return mlir::failure();

    // If embox/rebox and array_coor have conflicting shapes or slices,
    // do nothing.
    if (op.getShape() && boxedShape && boxedShape != op.getShape())
      return mlir::failure();
    if (op.getSlice() && boxedSlice && boxedSlice != op.getSlice())
      return mlir::failure();

    std::optional<IndicesVectorTy> shiftedIndices;
    // The embox/rebox and array_coor either have compatible
    // shape/slice at this point or shape/slice is null
    // in one of them but not in the other.
    // The compatibility means they are equal or both null.
    if (!op.getShape()) {
      if (boxedShape) {
        if (op.getSlice()) {
          if (!boxedSlice) {
            if (boxedShapeIsShift) {
              // %0 = fir.rebox %arg(%shift)
              // %1 = fir.array_coor %0 [%slice] %idx
              // Both the slice indices and %idx are 1-based, so the rebox
              // may be pulled in as:
              // %1 = fir.array_coor %arg [%slice] %idx
              boxedShape = nullptr;
            } else if (boxedShapeIsShape) {
              // %0 = fir.embox %arg(%shape)
              // %1 = fir.array_coor %0 [%slice] %idx
              // Pull in as:
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
            } else if (boxedShapeIsShapeShift) {
              // %0 = fir.embox %arg(%shapeshift)
              // %1 = fir.array_coor %0 [%slice] %idx
              // Pull in as:
              // %shape = fir.shape <extents from the %shapeshift>
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
              boxedShape = getShapeFromShapeShift(boxedShape, rewriter);
              if (!boxedShape)
                return mlir::failure();
            } else {
              return mlir::failure();
            }
          } else {
            if (boxedShapeIsShift) {
              // %0 = fir.rebox %arg(%shift) [%slice]
              // %1 = fir.array_coor %0 [%slice] %idx
              // This FIR may only be valid if the shape specifies
              // that all lower bounds are 1s and the slice's start indices
              // and strides are all 1s.
              // We could pull in the rebox as:
              // %1 = fir.array_coor %arg [%slice] %idx
              // Do not do anything for the time being.
              return mlir::failure();
            } else if (boxedShapeIsShape) {
              // %0 = fir.embox %arg(%shape) [%slice]
              // %1 = fir.array_coor %0 [%slice] %idx
              // This FIR may only be valid if the slice's start indices
              // and strides are all 1s.
              // We could pull in the embox as:
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
              return mlir::failure();
            } else if (boxedShapeIsShapeShift) {
              // %0 = fir.embox %arg(%shapeshift) [%slice]
              // %1 = fir.array_coor %0 [%slice] %idx
              // This FIR may only be valid if the shape specifies
              // that all lower bounds are 1s and the slice's start indices
              // and strides are all 1s.
              // We could pull in the embox as:
              // %shape = fir.shape <extents from the %shapeshift>
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
              return mlir::failure();
            } else {
              return mlir::failure();
            }
          }
        } else { // !op.getSlice()
          if (!boxedSlice) {
            if (boxedShapeIsShift) {
              // %0 = fir.rebox %arg(%shift)
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %1 = fir.array_coor %arg %idx
              boxedShape = nullptr;
            } else if (boxedShapeIsShape) {
              // %0 = fir.embox %arg(%shape)
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %1 = fir.array_coor %arg(%shape) %idx
            } else if (boxedShapeIsShapeShift) {
              // %0 = fir.embox %arg(%shapeshift)
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %shape = fir.shape <extents from the %shapeshift>
              // %1 = fir.array_coor %arg(%shape) %idx
              boxedShape = getShapeFromShapeShift(boxedShape, rewriter);
              if (!boxedShape)
                return mlir::failure();
            } else {
              return mlir::failure();
            }
          } else {
            if (boxedShapeIsShift) {
              // %0 = fir.embox %arg(%shift) [%slice]
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %tmp = arith.addi %idx, %shift.origin
              // %idx_shifted = arith.subi %tmp, 1
              // %1 = fir.array_coor %arg(%shift) %[slice] %idx_shifted
              shiftedIndices =
                  getShiftedIndices(boxedShape, op.getIndices(), rewriter);
              if (!shiftedIndices)
                return mlir::failure();
            } else if (boxedShapeIsShape) {
              // %0 = fir.embox %arg(%shape) [%slice]
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %1 = fir.array_coor %arg(%shape) %[slice] %idx
            } else if (boxedShapeIsShapeShift) {
              // %0 = fir.embox %arg(%shapeshift) [%slice]
              // %1 = fir.array_coor %0 %idx
              // Pull in as:
              // %tmp = arith.addi %idx, %shapeshift.lb
              // %idx_shifted = arith.subi %tmp, 1
              // %1 = fir.array_coor %arg(%shapeshift) %[slice] %idx_shifted
              shiftedIndices =
                  getShiftedIndices(boxedShape, op.getIndices(), rewriter);
              if (!shiftedIndices)
                return mlir::failure();
            } else {
              return mlir::failure();
            }
          }
        }
      } else { // !boxedShape
        if (op.getSlice()) {
          if (!boxedSlice) {
            // %0 = fir.rebox %arg
            // %1 = fir.array_coor %0 [%slice] %idx
            // Pull in as:
            // %1 = fir.array_coor %arg [%slice] %idx
          } else {
            // %0 = fir.rebox %arg [%slice]
            // %1 = fir.array_coor %0 [%slice] %idx
            // This is a valid FIR iff the slice's lower bounds
            // and strides are all 1s.
            // Pull in as:
            // %1 = fir.array_coor %arg [%slice] %idx
          }
        } else { // !op.getSlice()
          if (!boxedSlice) {
            // %0 = fir.rebox %arg
            // %1 = fir.array_coor %0 %idx
            // Pull in as:
            // %1 = fir.array_coor %arg %idx
          } else {
            // %0 = fir.rebox %arg [%slice]
            // %1 = fir.array_coor %0 %idx
            // Pull in as:
            // %1 = fir.array_coor %arg [%slice] %idx
          }
        }
      }
    } else { // op.getShape()
      if (boxedShape) {
        // Check if pulling in non-default shape is correct.
        if (op.getSlice()) {
          if (!boxedSlice) {
            // %0 = fir.embox %arg(%shape)
            // %1 = fir.array_coor %0(%shape) [%slice] %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) [%slice] %idx
          } else {
            // %0 = fir.embox %arg(%shape) [%slice]
            // %1 = fir.array_coor %0(%shape) [%slice] %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) [%slice] %idx
          }
        } else { // !op.getSlice()
          if (!boxedSlice) {
            // %0 = fir.embox %arg(%shape)
            // %1 = fir.array_coor %0(%shape) %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) %idx
          } else {
            // %0 = fir.embox %arg(%shape) [%slice]
            // %1 = fir.array_coor %0(%shape) %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) [%slice] %idx
          }
        }
      } else { // !boxedShape
        if (op.getSlice()) {
          if (!boxedSlice) {
            // %0 = fir.rebox %arg
            // %1 = fir.array_coor %0(%shape) [%slice] %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) [%slice] %idx
          } else {
            // %0 = fir.rebox %arg [%slice]
            // %1 = fir.array_coor %0(%shape) [%slice] %idx
            return mlir::failure();
          }
        } else { // !op.getSlice()
          if (!boxedSlice) {
            // %0 = fir.rebox %arg
            // %1 = fir.array_coor %0(%shape) %idx
            // Pull in as:
            // %1 = fir.array_coor %arg(%shape) %idx
          } else {
            // %0 = fir.rebox %arg [%slice]
            // %1 = fir.array_coor %0(%shape) %idx
            // Cannot pull in without adjusting the slice indices.
            return mlir::failure();
          }
        }
      }
    }

    // TODO: temporarily avoid producing array_coor with the shape shift
    // and plain array reference (it seems to be a limitation of
    // ArrayCoorOp verifier).
    if (!mlir::isa<fir::BaseBoxType>(boxedMemref.getType())) {
      if (boxedShape) {
        if (mlir::isa<fir::ShiftType>(boxedShape.getType()))
          return mlir::failure();
      } else if (op.getShape() &&
                 mlir::isa<fir::ShiftType>(op.getShape().getType())) {
        return mlir::failure();
      }
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getMemrefMutable().assign(boxedMemref);
      if (boxedShape)
        op.getShapeMutable().assign(boxedShape);
      if (boxedSlice)
        op.getSliceMutable().assign(boxedSlice);
      if (shiftedIndices)
        op.getIndicesMutable().assign(*shiftedIndices);
    });
    return mlir::success();
  }

private:
  using IndicesVectorTy = std::vector<mlir::Value>;

  // If v is a shape_shift operation:
  //   fir.shape_shift %l1, %e1, %l2, %e2, ...
  // create:
  //   fir.shape %e1, %e2, ...
  static mlir::Value getShapeFromShapeShift(mlir::Value v,
                                            mlir::PatternRewriter &rewriter) {
    auto shapeShiftOp =
        mlir::dyn_cast_or_null<fir::ShapeShiftOp>(v.getDefiningOp());
    if (!shapeShiftOp)
      return nullptr;
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(shapeShiftOp);
    return rewriter.create<fir::ShapeOp>(shapeShiftOp.getLoc(),
                                         shapeShiftOp.getExtents());
  }

  static std::optional<IndicesVectorTy>
  getShiftedIndices(mlir::Value v, mlir::ValueRange indices,
                    mlir::PatternRewriter &rewriter) {
    auto insertAdjustments = [&](mlir::Operation *op, mlir::ValueRange lbs) {
      // Compute the shifted indices using the extended type.
      // Note that this can probably result in less efficient
      // MLIR and further LLVM IR due to the extra conversions.
      mlir::OpBuilder::InsertPoint savedIP = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();
      mlir::Type idxTy = rewriter.getIndexType();
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
          loc, idxTy, rewriter.getIndexAttr(1));
      rewriter.restoreInsertionPoint(savedIP);
      auto nsw = mlir::arith::IntegerOverflowFlags::nsw;

      IndicesVectorTy shiftedIndices;
      for (auto [lb, idx] : llvm::zip(lbs, indices)) {
        mlir::Value extLb = rewriter.create<fir::ConvertOp>(loc, idxTy, lb);
        mlir::Value extIdx = rewriter.create<fir::ConvertOp>(loc, idxTy, idx);
        mlir::Value add =
            rewriter.create<mlir::arith::AddIOp>(loc, extIdx, extLb, nsw);
        mlir::Value sub =
            rewriter.create<mlir::arith::SubIOp>(loc, add, one, nsw);
        shiftedIndices.push_back(sub);
      }

      return shiftedIndices;
    };

    if (auto shiftOp =
            mlir::dyn_cast_or_null<fir::ShiftOp>(v.getDefiningOp())) {
      return insertAdjustments(shiftOp.getOperation(), shiftOp.getOrigins());
    } else if (auto shapeShiftOp = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(
                   v.getDefiningOp())) {
      return insertAdjustments(shapeShiftOp.getOperation(),
                               shapeShiftOp.getOrigins());
    }

    return std::nullopt;
  }
};

void fir::ArrayCoorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // TODO: !fir.shape<1> operand may be removed from array_coor always.
  patterns.add<SimplifyArrayCoorOp>(context);
}

//===----------------------------------------------------------------------===//
// ArrayLoadOp
//===----------------------------------------------------------------------===//

static mlir::Type adjustedElementType(mlir::Type t) {
  if (auto ty = mlir::dyn_cast<fir::ReferenceType>(t)) {
    auto eleTy = ty.getEleTy();
    if (fir::isa_char(eleTy))
      return eleTy;
    if (fir::isa_derived(eleTy))
      return eleTy;
    if (mlir::isa<fir::SequenceType>(eleTy))
      return eleTy;
  }
  return t;
}

std::vector<mlir::Value> fir::ArrayLoadOp::getExtents() {
  if (auto sh = getShape())
    if (auto *op = sh.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeOp>(op)) {
        auto extents = shOp.getExtents();
        return {extents.begin(), extents.end()};
      }
      return mlir::cast<fir::ShapeShiftOp>(op).getExtents();
    }
  return {};
}

void fir::ArrayLoadOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getMemrefMutable(),
                       mlir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

llvm::LogicalResult fir::ArrayLoadOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = mlir::dyn_cast<fir::SequenceType>(eleTy);
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0u;
    if (auto s = mlir::dyn_cast<fir::ShapeType>(shapeTy)) {
      shapeTyRank = s.getRank();
    } else if (auto ss = mlir::dyn_cast<fir::ShapeShiftType>(shapeTy)) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = mlir::cast<fir::ShiftType>(shapeTy);
      shapeTyRank = s.getRank();
      if (!mlir::isa<fir::BaseBoxType>(getMemref().getType()))
        return emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return emitOpError("rank of dimension mismatched");
  }

  if (auto sliceOp = getSlice()) {
    if (auto sl = mlir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_load cannot take a slice with substring");
    if (auto sliceTy = mlir::dyn_cast<fir::SliceType>(sliceOp.getType()))
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }

  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayMergeStoreOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayMergeStoreOp::verify() {
  if (!mlir::isa<fir::ArrayLoadOp>(getOriginal().getDefiningOp()))
    return emitOpError("operand #0 must be result of a fir.array_load op");
  if (auto sl = getSlice()) {
    if (auto sliceOp =
            mlir::dyn_cast_or_null<fir::SliceOp>(sl.getDefiningOp())) {
      if (!sliceOp.getSubstr().empty())
        return emitOpError(
            "array_merge_store cannot take a slice with substring");
      if (!sliceOp.getFields().empty()) {
        // This is an intra-object merge, where the slice is projecting the
        // subfields that are to be overwritten by the merge operation.
        auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
        if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(eleTy)) {
          auto projTy =
              fir::applyPathToType(seqTy.getEleTy(), sliceOp.getFields());
          if (fir::unwrapSequenceType(getOriginal().getType()) != projTy)
            return emitOpError(
                "type of origin does not match sliced memref type");
          if (fir::unwrapSequenceType(getSequence().getType()) != projTy)
            return emitOpError(
                "type of sequence does not match sliced memref type");
          return mlir::success();
        }
        return emitOpError("referenced type is not an array");
      }
    }
    return mlir::success();
  }
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  if (getOriginal().getType() != eleTy)
    return emitOpError("type of origin does not match memref element type");
  if (getSequence().getType() != eleTy)
    return emitOpError("type of sequence does not match memref element type");
  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");
  return mlir::success();
}

void fir::ArrayMergeStoreOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getMemrefMutable(),
                       mlir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// ArrayFetchOp
//===----------------------------------------------------------------------===//

// Template function used for both array_fetch and array_update verification.
template <typename A>
mlir::Type validArraySubobject(A op) {
  auto ty = op.getSequence().getType();
  return fir::applyPathToType(ty, op.getIndices());
}

llvm::LogicalResult fir::ArrayFetchOp::verify() {
  auto arrTy = mlir::cast<fir::SequenceType>(getSequence().getType());
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(getElement().getType()) != arrTy.getEleTy())
    return emitOpError("return type does not match array");
  auto ty = validArraySubobject(*this);
  if (!ty || ty != ::adjustedElementType(getType()))
    return emitOpError("return type and/or indices do not type check");
  if (!mlir::isa<fir::ArrayLoadOp>(getSequence().getDefiningOp()))
    return emitOpError("argument #0 must be result of fir.array_load");
  if (!validTypeParams(arrTy, getTypeparams()))
    return emitOpError("invalid type parameters");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayAccessOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayAccessOp::verify() {
  auto arrTy = mlir::cast<fir::SequenceType>(getSequence().getType());
  std::size_t indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      getElement().getType() != fir::ReferenceType::get(arrTy.getEleTy()))
    return emitOpError("return type does not match array");
  mlir::Type ty = validArraySubobject(*this);
  if (!ty || fir::ReferenceType::get(ty) != getType())
    return emitOpError("return type and/or indices do not type check");
  if (!validTypeParams(arrTy, getTypeparams()))
    return emitOpError("invalid type parameters");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayUpdateOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayUpdateOp::verify() {
  if (fir::isa_ref_type(getMerge().getType()))
    return emitOpError("does not support reference type for merge");
  auto arrTy = mlir::cast<fir::SequenceType>(getSequence().getType());
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(getMerge().getType()) != arrTy.getEleTy())
    return emitOpError("merged value does not have element type");
  auto ty = validArraySubobject(*this);
  if (!ty || ty != ::adjustedElementType(getMerge().getType()))
    return emitOpError("merged value and/or indices do not type check");
  if (!validTypeParams(arrTy, getTypeparams()))
    return emitOpError("invalid type parameters");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayModifyOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayModifyOp::verify() {
  auto arrTy = mlir::cast<fir::SequenceType>(getSequence().getType());
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices must match array dimension");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxAddrOp
//===----------------------------------------------------------------------===//

void fir::BoxAddrOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &result, mlir::Value val) {
  mlir::Type type =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(val.getType())
          .Case<fir::BaseBoxType>([&](fir::BaseBoxType ty) -> mlir::Type {
            mlir::Type eleTy = ty.getEleTy();
            if (fir::isa_ref_type(eleTy))
              return eleTy;
            return fir::ReferenceType::get(eleTy);
          })
          .Case<fir::BoxCharType>([&](fir::BoxCharType ty) -> mlir::Type {
            return fir::ReferenceType::get(ty.getEleTy());
          })
          .Case<fir::BoxProcType>(
              [&](fir::BoxProcType ty) { return ty.getEleTy(); })
          .Default([&](const auto &) { return mlir::Type{}; });
  assert(type && "bad val type");
  build(builder, result, type, val);
}

mlir::OpFoldResult fir::BoxAddrOp::fold(FoldAdaptor adaptor) {
  if (auto *v = getVal().getDefiningOp()) {
    if (auto box = mlir::dyn_cast<fir::EmboxOp>(v)) {
      // Fold only if not sliced
      if (!box.getSlice() && box.getMemref().getType() == getType()) {
        propagateAttributes(getOperation(), box.getMemref().getDefiningOp());
        return box.getMemref();
      }
    }
    if (auto box = mlir::dyn_cast<fir::EmboxCharOp>(v))
      if (box.getMemref().getType() == getType())
        return box.getMemref();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxCharLenOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult fir::BoxCharLenOp::fold(FoldAdaptor adaptor) {
  if (auto v = getVal().getDefiningOp()) {
    if (auto box = mlir::dyn_cast<fir::EmboxCharOp>(v))
      return box.getLen();
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
// BoxRankOp
//===----------------------------------------------------------------------===//

void fir::BoxRankOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  mlir::OpOperand &inputBox = getBoxMutable();
  if (fir::isBoxAddress(inputBox.get().getType()))
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &inputBox,
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::CallOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

void fir::CallOp::print(mlir::OpAsmPrinter &p) {
  bool isDirect = getCallee().has_value();
  p << ' ';
  if (isDirect)
    p << *getCallee();
  else
    p << getOperand(0);
  p << '(' << (*this)->getOperands().drop_front(isDirect ? 0 : 1) << ')';

  // Print `proc_attrs<...>`, if present.
  fir::FortranProcedureFlagsEnumAttr procAttrs = getProcedureAttrsAttr();
  if (procAttrs &&
      procAttrs.getValue() != fir::FortranProcedureFlagsEnum::none) {
    p << ' ' << fir::FortranProcedureFlagsEnumAttr::getMnemonic();
    p.printStrippedAttrOrType(procAttrs);
  }

  // Print 'fastmath<...>' (if it has non-default value) before
  // any other attributes.
  mlir::arith::FastMathFlagsAttr fmfAttr = getFastmathAttr();
  if (fmfAttr.getValue() != mlir::arith::FastMathFlags::none) {
    p << ' ' << mlir::arith::FastMathFlagsAttr::getMnemonic();
    p.printStrippedAttrOrType(fmfAttr);
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {fir::CallOp::getCalleeAttrNameStr(),
                           getFastmathAttrName(), getProcedureAttrsAttrName(),
                           getArgAttrsAttrName(), getResAttrsAttrName()});
  p << " : ";
  mlir::call_interface_impl::printFunctionSignature(
      p, getArgs().drop_front(isDirect ? 0 : 1).getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

mlir::ParseResult fir::CallOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return mlir::failure();

  mlir::NamedAttrList attrs;
  mlir::SymbolRefAttr funcAttr;
  bool isDirect = operands.empty();
  if (isDirect)
    if (parser.parseAttribute(funcAttr, fir::CallOp::getCalleeAttrNameStr(),
                              attrs))
      return mlir::failure();

  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  // Parse `proc_attrs<...>`, if present.
  fir::FortranProcedureFlagsEnumAttr procAttr;
  if (mlir::succeeded(parser.parseOptionalKeyword(
          fir::FortranProcedureFlagsEnumAttr::getMnemonic())))
    if (parser.parseCustomAttributeWithFallback(
            procAttr, mlir::Type{}, getProcedureAttrsAttrName(result.name),
            attrs))
      return mlir::failure();

  // Parse 'fastmath<...>', if present.
  mlir::arith::FastMathFlagsAttr fmfAttr;
  llvm::StringRef fmfAttrName = getFastmathAttrName(result.name);
  if (mlir::succeeded(parser.parseOptionalKeyword(fmfAttrName)))
    if (parser.parseCustomAttributeWithFallback(fmfAttr, mlir::Type{},
                                                fmfAttrName, attrs))
      return mlir::failure();

  if (parser.parseOptionalAttrDict(attrs) || parser.parseColon())
    return mlir::failure();
  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resTypes;
  llvm::SmallVector<mlir::DictionaryAttr> argAttrs;
  llvm::SmallVector<mlir::DictionaryAttr> resultAttrs;
  if (mlir::call_interface_impl::parseFunctionSignature(
          parser, argTypes, argAttrs, resTypes, resultAttrs))
    return parser.emitError(parser.getNameLoc(), "expected function type");
  mlir::FunctionType funcType =
      mlir::FunctionType::get(parser.getContext(), argTypes, resTypes);
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand>(operands)
            .drop_front();
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  }
  result.attributes = attrs;
  mlir::call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  result.addTypes(funcType.getResults());
  return mlir::success();
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::func::FuncOp callee, mlir::ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrNameStr(), mlir::SymbolRefAttr::get(callee));
  result.addTypes(callee.getFunctionType().getResults());
}

void fir::CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::SymbolRefAttr callee,
                        llvm::ArrayRef<mlir::Type> results,
                        mlir::ValueRange operands) {
  result.addOperands(operands);
  if (callee)
    result.addAttribute(getCalleeAttrNameStr(), callee);
  result.addTypes(results);
}

//===----------------------------------------------------------------------===//
// CharConvertOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::CharConvertOp::verify() {
  auto unwrap = [&](mlir::Type t) {
    t = fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t));
    return mlir::dyn_cast<fir::CharacterType>(t);
  };
  auto inTy = unwrap(getFrom().getType());
  auto outTy = unwrap(getTo().getType());
  if (!(inTy && outTy))
    return emitOpError("not a reference to a character");
  if (inTy.getFKind() == outTy.getFKind())
    return emitOpError("buffers must have different KIND values");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

template <typename OPTY>
static void printCmpOp(mlir::OpAsmPrinter &p, OPTY op) {
  p << ' ';
  auto predSym = mlir::arith::symbolizeCmpFPredicate(
      op->template getAttrOfType<mlir::IntegerAttr>(
            OPTY::getPredicateAttrName())
          .getInt());
  assert(predSym.has_value() && "invalid symbol value for predicate");
  p << '"' << mlir::arith::stringifyCmpFPredicate(predSym.value()) << '"'
    << ", ";
  p.printOperand(op.getLhs());
  p << ", ";
  p.printOperand(op.getRhs());
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.getLhs().getType();
}

template <typename OPTY>
static mlir::ParseResult parseCmpOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> ops;
  mlir::NamedAttrList attrs;
  mlir::Attribute predicateNameAttr;
  mlir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return mlir::failure();

  if (!mlir::isa<mlir::StringAttr>(predicateNameAttr))
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      mlir::cast<mlir::StringAttr>(predicateNameAttr).getValue();
  auto predicate = fir::CmpcOp::getPredicateByName(predicateName);
  auto builder = parser.getBuilder();
  mlir::Type i1Type = builder.getI1Type();
  attrs.set(OPTY::getPredicateAttrName(),
            builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CmpcOp
//===----------------------------------------------------------------------===//

void fir::buildCmpCOp(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::arith::CmpFPredicate predicate, mlir::Value lhs,
                      mlir::Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      fir::CmpcOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
}

mlir::arith::CmpFPredicate
fir::CmpcOp::getPredicateByName(llvm::StringRef name) {
  auto pred = mlir::arith::symbolizeCmpFPredicate(name);
  assert(pred.has_value() && "invalid predicate name");
  return pred.value();
}

void fir::CmpcOp::print(mlir::OpAsmPrinter &p) { printCmpOp(p, *this); }

mlir::ParseResult fir::CmpcOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// VolatileCastOp
//===----------------------------------------------------------------------===//

static bool typesMatchExceptForVolatility(mlir::Type fromType,
                                          mlir::Type toType) {
  // If we can change only the volatility and get identical types, then we
  // match.
  if (fir::updateTypeWithVolatility(fromType, fir::isa_volatile_type(toType)) ==
      toType)
    return true;

  // Otherwise, recurse on the element types if the base classes are the same.
  const bool match =
      llvm::TypeSwitch<mlir::Type, bool>(fromType)
          .Case<fir::BoxType, fir::ReferenceType, fir::ClassType>(
              [&](auto type) {
                using TYPE = decltype(type);
                // If we are not the same base class, then we don't match.
                auto castedToType = mlir::dyn_cast<TYPE>(toType);
                if (!castedToType)
                  return false;
                // If we are the same base class, we match if the element types
                // match.
                return typesMatchExceptForVolatility(type.getEleTy(),
                                                     castedToType.getEleTy());
              })
          .Default([](mlir::Type) { return false; });

  return match;
}

llvm::LogicalResult fir::VolatileCastOp::verify() {
  mlir::Type fromType = getValue().getType();
  mlir::Type toType = getType();
  if (!typesMatchExceptForVolatility(fromType, toType))
    return emitOpError("types must be identical except for volatility ")
           << fromType << " / " << toType;
  return mlir::success();
}

mlir::OpFoldResult fir::VolatileCastOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getType())
    return getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void fir::ConvertOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<ConvertConvertOptPattern, ConvertAscendingIndexOptPattern,
                 ConvertDescendingIndexOptPattern, RedundantConvertOptPattern,
                 CombineConvertOptPattern, CombineConvertTruncOptPattern,
                 ForwardConstantConvertPattern, ChainedPointerConvertsPattern>(
      context);
}

mlir::OpFoldResult fir::ConvertOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getType())
    return getValue();
  if (matchPattern(getValue(), mlir::m_Op<fir::ConvertOp>())) {
    auto inner = mlir::cast<fir::ConvertOp>(getValue().getDefiningOp());
    // (convert (convert 'a : logical -> i1) : i1 -> logical) ==> forward 'a
    if (auto toTy = mlir::dyn_cast<fir::LogicalType>(getType()))
      if (auto fromTy =
              mlir::dyn_cast<fir::LogicalType>(inner.getValue().getType()))
        if (mlir::isa<mlir::IntegerType>(inner.getType()) && (toTy == fromTy))
          return inner.getValue();
    // (convert (convert 'a : i1 -> logical) : logical -> i1) ==> forward 'a
    if (auto toTy = mlir::dyn_cast<mlir::IntegerType>(getType()))
      if (auto fromTy =
              mlir::dyn_cast<mlir::IntegerType>(inner.getValue().getType()))
        if (mlir::isa<fir::LogicalType>(inner.getType()) && (toTy == fromTy) &&
            (fromTy.getWidth() == 1))
          return inner.getValue();
  }
  return {};
}

bool fir::ConvertOp::isInteger(mlir::Type ty) {
  return mlir::isa<mlir::IntegerType, mlir::IndexType, fir::IntegerType>(ty);
}

bool fir::ConvertOp::isIntegerCompatible(mlir::Type ty) {
  return isInteger(ty) || mlir::isa<fir::LogicalType>(ty);
}

bool fir::ConvertOp::isFloatCompatible(mlir::Type ty) {
  return mlir::isa<mlir::FloatType>(ty);
}

bool fir::ConvertOp::isPointerCompatible(mlir::Type ty) {
  return mlir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
                   fir::LLVMPointerType, mlir::MemRefType, mlir::FunctionType,
                   fir::TypeDescType, mlir::LLVM::LLVMPointerType>(ty);
}

static std::optional<mlir::Type> getVectorElementType(mlir::Type ty) {
  mlir::Type elemTy;
  if (mlir::isa<fir::VectorType>(ty))
    elemTy = mlir::dyn_cast<fir::VectorType>(ty).getElementType();
  else if (mlir::isa<mlir::VectorType>(ty))
    elemTy = mlir::dyn_cast<mlir::VectorType>(ty).getElementType();
  else
    return std::nullopt;

  // e.g. fir.vector<4:ui32> => mlir.vector<4xi32>
  // e.g. mlir.vector<4xui32> => mlir.vector<4xi32>
  if (elemTy.isUnsignedInteger()) {
    elemTy = mlir::IntegerType::get(
        ty.getContext(), mlir::dyn_cast<mlir::IntegerType>(elemTy).getWidth());
  }
  return elemTy;
}

static std::optional<uint64_t> getVectorLen(mlir::Type ty) {
  if (mlir::isa<fir::VectorType>(ty))
    return mlir::dyn_cast<fir::VectorType>(ty).getLen();
  else if (mlir::isa<mlir::VectorType>(ty)) {
    // fir.vector only supports 1-D vector
    if (!(mlir::dyn_cast<mlir::VectorType>(ty).isScalable()))
      return mlir::dyn_cast<mlir::VectorType>(ty).getShape()[0];
  }

  return std::nullopt;
}

bool fir::ConvertOp::areVectorsCompatible(mlir::Type inTy, mlir::Type outTy) {
  if (!(mlir::isa<fir::VectorType>(inTy) &&
        mlir::isa<mlir::VectorType>(outTy)) &&
      !(mlir::isa<mlir::VectorType>(inTy) && mlir::isa<fir::VectorType>(outTy)))
    return false;

  // Only support integer, unsigned and real vector
  // Both vectors must have the same element type
  std::optional<mlir::Type> inElemTy = getVectorElementType(inTy);
  std::optional<mlir::Type> outElemTy = getVectorElementType(outTy);
  if (!inElemTy.has_value() || !outElemTy.has_value() ||
      inElemTy.value() != outElemTy.value())
    return false;

  // Both vectors must have the same number of elements
  std::optional<uint64_t> inLen = getVectorLen(inTy);
  std::optional<uint64_t> outLen = getVectorLen(outTy);
  if (!inLen.has_value() || !outLen.has_value() ||
      inLen.value() != outLen.value())
    return false;

  return true;
}

static bool areRecordsCompatible(mlir::Type inTy, mlir::Type outTy) {
  // Both records must have the same field types.
  // Trust frontend semantics for in-depth checks, such as if both records
  // have the BIND(C) attribute.
  auto inRecTy = mlir::dyn_cast<fir::RecordType>(inTy);
  auto outRecTy = mlir::dyn_cast<fir::RecordType>(outTy);
  return inRecTy && outRecTy && inRecTy.getTypeList() == outRecTy.getTypeList();
}

bool fir::ConvertOp::canBeConverted(mlir::Type inType, mlir::Type outType) {
  if (inType == outType)
    return true;
  return (isPointerCompatible(inType) && isPointerCompatible(outType)) ||
         (isIntegerCompatible(inType) && isIntegerCompatible(outType)) ||
         (isInteger(inType) && isFloatCompatible(outType)) ||
         (isFloatCompatible(inType) && isInteger(outType)) ||
         (isFloatCompatible(inType) && isFloatCompatible(outType)) ||
         (isIntegerCompatible(inType) && isPointerCompatible(outType)) ||
         (isPointerCompatible(inType) && isIntegerCompatible(outType)) ||
         (mlir::isa<fir::BoxType>(inType) &&
          mlir::isa<fir::BoxType>(outType)) ||
         (mlir::isa<fir::BoxProcType>(inType) &&
          mlir::isa<fir::BoxProcType>(outType)) ||
         (fir::isa_complex(inType) && fir::isa_complex(outType)) ||
         (fir::isBoxedRecordType(inType) && fir::isPolymorphicType(outType)) ||
         (fir::isPolymorphicType(inType) && fir::isPolymorphicType(outType)) ||
         (fir::isPolymorphicType(inType) && mlir::isa<BoxType>(outType)) ||
         areVectorsCompatible(inType, outType) ||
         areRecordsCompatible(inType, outType);
}

// In general, ptrtoint-like conversions are allowed to lose volatility
// information because they are either:
//
// 1. passing an entity to an external function and there's nothing we can do
//    about volatility after that happens, or
// 2. for code generation, at which point we represent volatility with
//    attributes on the LLVM instructions and intrinsics.
//
// For all other cases, volatility ought to match exactly.
static mlir::LogicalResult verifyVolatility(mlir::Type inType,
                                            mlir::Type outType) {
  const bool toLLVMPointer = mlir::isa<mlir::LLVM::LLVMPointerType>(outType);
  const bool toInteger = fir::isa_integer(outType);

  // When converting references to classes or allocatables into boxes for
  // runtime arguments, we cast away all the volatility information and pass a
  // box<none>. This is allowed.
  const bool isBoxNoneLike = [&]() {
    if (fir::isBoxNone(outType))
      return true;
    if (auto referenceType = mlir::dyn_cast<fir::ReferenceType>(outType)) {
      if (fir::isBoxNone(referenceType.getElementType())) {
        return true;
      }
    }
    return false;
  }();

  const bool isPtrToIntLike = toLLVMPointer || toInteger || isBoxNoneLike;
  if (isPtrToIntLike) {
    return mlir::success();
  }

  // In all other cases, we need to check for an exact volatility match.
  return mlir::success(fir::isa_volatile_type(inType) ==
                       fir::isa_volatile_type(outType));
}

llvm::LogicalResult fir::ConvertOp::verify() {
  mlir::Type inType = getValue().getType();
  mlir::Type outType = getType();
  if (fir::useStrictVolatileVerification()) {
    if (failed(verifyVolatility(inType, outType))) {
      return emitOpError("this conversion does not preserve volatility: ")
             << inType << " / " << outType;
    }
  }
  if (canBeConverted(inType, outType))
    return mlir::success();
  return emitOpError("invalid type conversion")
         << getValue().getType() << " / " << getType();
}

//===----------------------------------------------------------------------===//
// CoordinateOp
//===----------------------------------------------------------------------===//

void fir::CoordinateOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Type resultType, mlir::Value ref,
                              mlir::ValueRange coor) {
  llvm::SmallVector<int32_t> fieldIndices;
  llvm::SmallVector<mlir::Value> dynamicIndices;
  bool anyField = false;
  for (mlir::Value index : coor) {
    if (auto field = index.getDefiningOp<fir::FieldIndexOp>()) {
      auto recTy = mlir::cast<fir::RecordType>(field.getOnType());
      fieldIndices.push_back(recTy.getFieldIndex(field.getFieldId()));
      anyField = true;
    } else {
      fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
      dynamicIndices.push_back(index);
    }
  }
  auto typeAttr = mlir::TypeAttr::get(ref.getType());
  if (anyField) {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr,
          builder.getDenseI32ArrayAttr(fieldIndices));
  } else {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr, nullptr);
  }
}

void fir::CoordinateOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Type resultType, mlir::Value ref,
                              llvm::ArrayRef<fir::IntOrValue> coor) {
  llvm::SmallVector<int32_t> fieldIndices;
  llvm::SmallVector<mlir::Value> dynamicIndices;
  bool anyField = false;
  for (fir::IntOrValue index : coor) {
    llvm::TypeSwitch<fir::IntOrValue>(index)
        .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr intAttr) {
          fieldIndices.push_back(intAttr.getInt());
          anyField = true;
        })
        .Case<mlir::Value>([&](mlir::Value value) {
          dynamicIndices.push_back(value);
          fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
        });
  }
  auto typeAttr = mlir::TypeAttr::get(ref.getType());
  if (anyField) {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr,
          builder.getDenseI32ArrayAttr(fieldIndices));
  } else {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr, nullptr);
  }
}

void fir::CoordinateOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getRef();
  if (!getFieldIndicesAttr()) {
    p << ", " << getCoor();
  } else {
    mlir::Type eleTy = fir::getFortranElementType(getRef().getType());
    for (auto index : getIndices()) {
      p << ", ";
      llvm::TypeSwitch<fir::IntOrValue>(index)
          .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr intAttr) {
            if (auto recordType = llvm::dyn_cast<fir::RecordType>(eleTy)) {
              int fieldId = intAttr.getInt();
              if (fieldId < static_cast<int>(recordType.getNumFields())) {
                auto nameAndType = recordType.getTypeList()[fieldId];
                p << std::get<std::string>(nameAndType);
                eleTy = fir::getFortranElementType(
                    std::get<mlir::Type>(nameAndType));
                return;
              }
            }
            // Invalid index, still print it so that invalid IR can be
            // investigated.
            p << intAttr;
          })
          .Case<mlir::Value>([&](mlir::Value value) { p << value; });
    }
  }
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elideAttrs=*/{getBaseTypeAttrName(), getFieldIndicesAttrName()});
  p << " : ";
  p.printFunctionalType(getOperandTypes(), (*this)->getResultTypes());
}

mlir::ParseResult fir::CoordinateOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand memref;
  if (parser.parseOperand(memref) || parser.parseComma())
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> coorOperands;
  llvm::SmallVector<std::pair<llvm::StringRef, int>> fieldNames;
  llvm::SmallVector<int32_t> fieldIndices;
  while (true) {
    llvm::StringRef fieldName;
    if (mlir::succeeded(parser.parseOptionalKeyword(&fieldName))) {
      fieldNames.push_back({fieldName, static_cast<int>(fieldIndices.size())});
      // Actual value will be computed later when base type has been parsed.
      fieldIndices.push_back(0);
    } else {
      mlir::OpAsmParser::UnresolvedOperand index;
      if (parser.parseOperand(index))
        return mlir::failure();
      fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
      coorOperands.push_back(index);
    }
    if (mlir::failed(parser.parseOptionalComma()))
      break;
  }
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> allOperands;
  allOperands.push_back(memref);
  allOperands.append(coorOperands.begin(), coorOperands.end());
  mlir::FunctionType funcTy;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcTy) ||
      parser.resolveOperands(allOperands, funcTy.getInputs(), loc,
                             result.operands) ||
      parser.addTypesToList(funcTy.getResults(), result.types))
    return mlir::failure();
  result.addAttribute(getBaseTypeAttrName(result.name),
                      mlir::TypeAttr::get(funcTy.getInput(0)));
  if (!fieldNames.empty()) {
    mlir::Type eleTy = fir::getFortranElementType(funcTy.getInput(0));
    for (auto [fieldName, operandPosition] : fieldNames) {
      auto recTy = llvm::dyn_cast<fir::RecordType>(eleTy);
      if (!recTy)
        return parser.emitError(
            loc, "base must be a derived type when field name appears");
      unsigned fieldNum = recTy.getFieldIndex(fieldName);
      if (fieldNum > recTy.getNumFields())
        return parser.emitError(loc)
               << "field '" << fieldName
               << "' is not a component or subcomponent of the base type";
      fieldIndices[operandPosition] = fieldNum;
      eleTy = fir::getFortranElementType(
          std::get<mlir::Type>(recTy.getTypeList()[fieldNum]));
    }
    result.addAttribute(getFieldIndicesAttrName(result.name),
                        parser.getBuilder().getDenseI32ArrayAttr(fieldIndices));
  }
  return mlir::success();
}

llvm::LogicalResult fir::CoordinateOp::verify() {
  const mlir::Type refTy = getRef().getType();
  if (fir::isa_ref_type(refTy)) {
    auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
    if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(eleTy)) {
      if (arrTy.hasUnknownShape())
        return emitOpError("cannot find coordinate in unknown shape");
      if (arrTy.getConstantRows() < arrTy.getDimension() - 1)
        return emitOpError("cannot find coordinate with unknown extents");
    }
    if (!(fir::isa_aggregate(eleTy) || fir::isa_complex(eleTy) ||
          fir::isa_char_string(eleTy)))
      return emitOpError("cannot apply to this element type");
  }
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(refTy);
  unsigned dimension = 0;
  const unsigned numCoors = getCoor().size();
  for (auto coorOperand : llvm::enumerate(getCoor())) {
    auto co = coorOperand.value();
    if (dimension == 0 && mlir::isa<fir::SequenceType>(eleTy)) {
      dimension = mlir::cast<fir::SequenceType>(eleTy).getDimension();
      if (dimension == 0)
        return emitOpError("cannot apply to array of unknown rank");
    }
    if (auto *defOp = co.getDefiningOp()) {
      if (auto index = mlir::dyn_cast<fir::LenParamIndexOp>(defOp)) {
        // Recovering a LEN type parameter only makes sense from a boxed
        // value. For a bare reference, the LEN type parameters must be
        // passed as additional arguments to `index`.
        if (mlir::isa<fir::BoxType>(refTy)) {
          if (coorOperand.index() != numCoors - 1)
            return emitOpError("len_param_index must be last argument");
          if (getNumOperands() != 2)
            return emitOpError("too many operands for len_param_index case");
        }
        if (eleTy != index.getOnType())
          emitOpError(
              "len_param_index type not compatible with reference type");
        return mlir::success();
      } else if (auto index = mlir::dyn_cast<fir::FieldIndexOp>(defOp)) {
        if (eleTy != index.getOnType())
          emitOpError("field_index type not compatible with reference type");
        if (auto recTy = mlir::dyn_cast<fir::RecordType>(eleTy)) {
          eleTy = recTy.getType(index.getFieldName());
          continue;
        }
        return emitOpError("field_index not applied to !fir.type");
      }
    }
    if (dimension) {
      if (--dimension == 0)
        eleTy = mlir::cast<fir::SequenceType>(eleTy).getElementType();
    } else {
      if (auto t = mlir::dyn_cast<mlir::TupleType>(eleTy)) {
        // FIXME: Generally, we don't know which field of the tuple is being
        // referred to unless the operand is a constant. Just assume everything
        // is good in the tuple case for now.
        return mlir::success();
      } else if (auto t = mlir::dyn_cast<fir::RecordType>(eleTy)) {
        // FIXME: This is the same as the tuple case.
        return mlir::success();
      } else if (auto t = mlir::dyn_cast<mlir::ComplexType>(eleTy)) {
        eleTy = t.getElementType();
      } else if (auto t = mlir::dyn_cast<fir::CharacterType>(eleTy)) {
        if (t.getLen() == fir::CharacterType::singleton())
          return emitOpError("cannot apply to character singleton");
        eleTy = fir::CharacterType::getSingleton(t.getContext(), t.getFKind());
        if (fir::unwrapRefType(getType()) != eleTy)
          return emitOpError("character type mismatch");
      } else {
        return emitOpError("invalid parameters (too many)");
      }
    }
  }
  return mlir::success();
}

fir::CoordinateIndicesAdaptor fir::CoordinateOp::getIndices() {
  return CoordinateIndicesAdaptor(getFieldIndicesAttr(), getCoor());
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::DispatchOp::verify() {
  // Check that pass_arg_pos is in range of actual operands. pass_arg_pos is
  // unsigned so check for less than zero is not needed.
  if (getPassArgPos() && *getPassArgPos() > (getArgOperands().size() - 1))
    return emitOpError(
        "pass_arg_pos must be smaller than the number of operands");

  // Operand pointed by pass_arg_pos must have polymorphic type.
  if (getPassArgPos() &&
      !fir::isPolymorphicType(getArgOperands()[*getPassArgPos()].getType()))
    return emitOpError("pass_arg_pos must be a polymorphic operand");
  return mlir::success();
}

mlir::FunctionType fir::DispatchOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

//===----------------------------------------------------------------------===//
// TypeInfoOp
//===----------------------------------------------------------------------===//

void fir::TypeInfoOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, fir::RecordType type,
                            fir::RecordType parentType,
                            llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  result.addRegion();
  result.addRegion();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(type.getName()));
  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));
  if (parentType)
    result.addAttribute(getParentTypeAttrName(result.name),
                        mlir::TypeAttr::get(parentType));
  result.addAttributes(attrs);
}

llvm::LogicalResult fir::TypeInfoOp::verify() {
  if (!getDispatchTable().empty())
    for (auto &op : getDispatchTable().front().without_terminator())
      if (!mlir::isa<fir::DTEntryOp>(op))
        return op.emitOpError("dispatch table must contain dt_entry");

  if (!mlir::isa<fir::RecordType>(getType()))
    return emitOpError("type must be a fir.type");

  if (getParentType() && !mlir::isa<fir::RecordType>(*getParentType()))
    return emitOpError("parent_type must be a fir.type");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxOp
//===----------------------------------------------------------------------===//

// Conversions from reference types to box types must preserve volatility.
static llvm::LogicalResult
verifyEmboxOpVolatilityInvariants(mlir::Type memrefType,
                                  mlir::Type resultType) {

  if (!fir::useStrictVolatileVerification())
    return mlir::success();

  mlir::Type boxElementType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(resultType)
          .Case<fir::BoxType, fir::ClassType>(
              [&](auto type) { return type.getEleTy(); })
          .Default([&](mlir::Type type) { return type; });

  // If the embox is simply wrapping a non-volatile type into a volatile box,
  // we're not losing any volatility information.
  if (boxElementType == memrefType) {
    return mlir::success();
  }

  // Otherwise, the volatility of the input and result must match.
  const bool volatilityMatches =
      fir::isa_volatile_type(memrefType) == fir::isa_volatile_type(resultType);

  return mlir::success(volatilityMatches);
}

llvm::LogicalResult fir::EmboxOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  bool isArray = false;
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(eleTy)) {
    eleTy = seqTy.getEleTy();
    isArray = true;
  }
  if (hasLenParams()) {
    auto lenPs = numLenParams();
    if (auto rt = mlir::dyn_cast<fir::RecordType>(eleTy)) {
      if (lenPs != rt.getNumLenParams())
        return emitOpError("number of LEN params does not correspond"
                           " to the !fir.type type");
    } else if (auto strTy = mlir::dyn_cast<fir::CharacterType>(eleTy)) {
      if (strTy.getLen() != fir::CharacterType::unknownLen())
        return emitOpError("CHARACTER already has static LEN");
    } else {
      return emitOpError("LEN parameters require CHARACTER or derived type");
    }
    for (auto lp : getTypeparams())
      if (!fir::isa_integer(lp.getType()))
        return emitOpError("LEN parameters must be integral type");
  }
  if (getShape() && !isArray)
    return emitOpError("shape must not be provided for a scalar");
  if (getSlice() && !isArray)
    return emitOpError("slice must not be provided for a scalar");
  if (getSourceBox() && !mlir::isa<fir::ClassType>(getResult().getType()))
    return emitOpError("source_box must be used with fir.class result type");
  if (failed(verifyEmboxOpVolatilityInvariants(getMemref().getType(),
                                               getResult().getType())))
    return emitOpError(
               "cannot convert between volatile and non-volatile types:")
           << " " << getMemref().getType() << " " << getResult().getType();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxCharOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::EmboxCharOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  if (!mlir::dyn_cast_or_null<fir::CharacterType>(eleTy))
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EmboxProcOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::EmboxProcOp::verify() {
  // host bindings (optional) must be a reference to a tuple
  if (auto h = getHost()) {
    if (auto r = mlir::dyn_cast<fir::ReferenceType>(h.getType()))
      if (mlir::isa<mlir::TupleType>(r.getEleTy()))
        return mlir::success();
    return mlir::failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TypeDescOp
//===----------------------------------------------------------------------===//

void fir::TypeDescOp::build(mlir::OpBuilder &, mlir::OperationState &result,
                            mlir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

mlir::ParseResult fir::TypeDescOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::Type intype;
  if (parser.parseType(intype))
    return mlir::failure();
  result.addAttribute("in_type", mlir::TypeAttr::get(intype));
  mlir::Type restype = fir::TypeDescType::get(intype);
  if (parser.addTypeToList(restype, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::TypeDescOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr("in_type");
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"in_type"});
}

llvm::LogicalResult fir::TypeDescOp::verify() {
  mlir::Type resultTy = getType();
  if (auto tdesc = mlir::dyn_cast<fir::TypeDescType>(resultTy)) {
    if (tdesc.getOfTy() != getInType())
      return emitOpError("wrapped type mismatched");
    return mlir::success();
  }
  return emitOpError("must be !fir.tdesc type");
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

mlir::Type fir::GlobalOp::resultType() {
  return wrapAllocaResultType(getType());
}

mlir::ParseResult fir::GlobalOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  // Parse the optional linkage
  llvm::StringRef linkage;
  auto &builder = parser.getBuilder();
  if (mlir::succeeded(parser.parseOptionalKeyword(&linkage))) {
    if (fir::GlobalOp::verifyValidLinkage(linkage))
      return mlir::failure();
    mlir::StringAttr linkAttr = builder.getStringAttr(linkage);
    result.addAttribute(fir::GlobalOp::getLinkNameAttrName(result.name),
                        linkAttr);
  }

  // Parse the name as a symbol reference attribute.
  mlir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr,
                            fir::GlobalOp::getSymrefAttrName(result.name),
                            result.attributes))
    return mlir::failure();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      nameAttr.getRootReference());

  bool simpleInitializer = false;
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    mlir::Attribute attr;
    if (parser.parseAttribute(attr, getInitValAttrName(result.name),
                              result.attributes) ||
        parser.parseRParen())
      return mlir::failure();
    simpleInitializer = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  if (succeeded(
          parser.parseOptionalKeyword(getConstantAttrName(result.name)))) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  }

  if (succeeded(parser.parseOptionalKeyword(getTargetAttrName(result.name))))
    result.addAttribute(getTargetAttrName(result.name), builder.getUnitAttr());

  mlir::Type globalType;
  if (parser.parseColonType(globalType))
    return mlir::failure();

  result.addAttribute(fir::GlobalOp::getTypeAttrName(result.name),
                      mlir::TypeAttr::get(globalType));

  if (simpleInitializer) {
    result.addRegion();
  } else {
    // Parse the optional initializer body.
    auto parseResult =
        parser.parseOptionalRegion(*result.addRegion(), /*arguments=*/{});
    if (parseResult.has_value() && mlir::failed(*parseResult))
      return mlir::failure();
  }
  return mlir::success();
}

void fir::GlobalOp::print(mlir::OpAsmPrinter &p) {
  if (getLinkName())
    p << ' ' << *getLinkName();
  p << ' ';
  p.printAttributeWithoutType(getSymrefAttr());
  if (auto val = getValueOrNull())
    p << '(' << val << ')';
  // Print all other attributes that are not pretty printed here.
  p.printOptionalAttrDict((*this)->getAttrs(), /*elideAttrs=*/{
                              getSymNameAttrName(), getSymrefAttrName(),
                              getTypeAttrName(), getConstantAttrName(),
                              getTargetAttrName(), getLinkNameAttrName(),
                              getInitValAttrName()});
  if (getOperation()->getAttr(getConstantAttrName()))
    p << " " << getConstantAttrName().strref();
  if (getOperation()->getAttr(getTargetAttrName()))
    p << " " << getTargetAttrName().strref();
  p << " : ";
  p.printType(getType());
  if (hasInitializationBody()) {
    p << ' ';
    p.printRegion(getOperation()->getRegion(0),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

void fir::GlobalOp::appendInitialValue(mlir::Operation *op) {
  getBlock().getOperations().push_back(op);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, mlir::Type type,
                          mlir::Attribute initialVal, mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(getTypeAttrName(result.name), mlir::TypeAttr::get(type));
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getSymrefAttrName(result.name),
                      mlir::SymbolRefAttr::get(builder.getContext(), name));
  if (isConstant)
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  if (isTarget)
    result.addAttribute(getTargetAttrName(result.name), builder.getUnitAttr());
  if (initialVal)
    result.addAttribute(getInitValAttrName(result.name), initialVal);
  if (linkage)
    result.addAttribute(getLinkNameAttrName(result.name), linkage);
  result.attributes.append(attrs.begin(), attrs.end());
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type, mlir::Attribute initialVal,
                          mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, mlir::Type type,
                          mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, isTarget, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type, mlir::StringAttr linkage,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, mlir::Type type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, isTarget, type, mlir::StringAttr{},
        attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, llvm::StringRef name,
                          mlir::Type type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        attrs);
}

mlir::ParseResult fir::GlobalOp::verifyValidLinkage(llvm::StringRef linkage) {
  // Supporting only a subset of the LLVM linkage types for now
  static const char *validNames[] = {"common", "internal", "linkonce",
                                     "linkonce_odr", "weak"};
  return mlir::success(llvm::is_contained(validNames, linkage));
}

//===----------------------------------------------------------------------===//
// GlobalLenOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::GlobalLenOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  llvm::StringRef fieldName;
  if (failed(parser.parseOptionalKeyword(&fieldName))) {
    mlir::StringAttr fieldAttr;
    if (parser.parseAttribute(fieldAttr,
                              fir::GlobalLenOp::getLenParamAttrName(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(fir::GlobalLenOp::getLenParamAttrName(),
                        parser.getBuilder().getStringAttr(fieldName));
  }
  mlir::IntegerAttr constant;
  if (parser.parseComma() ||
      parser.parseAttribute(constant, fir::GlobalLenOp::getIntAttrName(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

void fir::GlobalLenOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr(fir::GlobalLenOp::getLenParamAttrName())
    << ", " << getOperation()->getAttr(fir::GlobalLenOp::getIntAttrName());
}

//===----------------------------------------------------------------------===//
// FieldIndexOp
//===----------------------------------------------------------------------===//

template <typename TY>
mlir::ParseResult parseFieldLikeOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  mlir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::getFieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!mlir::dyn_cast<fir::RecordType>(recty))
    return mlir::failure();
  result.addAttribute(fir::FieldIndexOp::getTypeAttrName(),
                      mlir::TypeAttr::get(recty));
  if (!parser.parseOptionalLParen()) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> types;
    auto loc = parser.getNameLoc();
    if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(types) || parser.parseRParen() ||
        parser.resolveOperands(operands, types, loc, result.operands))
      return mlir::failure();
  }
  mlir::Type fieldType = TY::get(builder.getContext());
  if (parser.addTypeToList(fieldType, result.types))
    return mlir::failure();
  return mlir::success();
}

mlir::ParseResult fir::FieldIndexOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  return parseFieldLikeOp<fir::FieldType>(parser, result);
}

template <typename OP>
void printFieldLikeOp(mlir::OpAsmPrinter &p, OP &op) {
  p << ' '
    << op.getOperation()
           ->template getAttrOfType<mlir::StringAttr>(
               fir::FieldIndexOp::getFieldAttrName())
           .getValue()
    << ", " << op.getOperation()->getAttr(fir::FieldIndexOp::getTypeAttrName());
  if (op.getNumOperands()) {
    p << '(';
    p.printOperands(op.getTypeparams());
    auto sep = ") : ";
    for (auto op : op.getTypeparams()) {
      p << sep;
      if (op)
        p.printType(op.getType());
      else
        p << "()";
      sep = ", ";
    }
  }
}

void fir::FieldIndexOp::print(mlir::OpAsmPrinter &p) {
  printFieldLikeOp(p, *this);
}

void fir::FieldIndexOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              llvm::StringRef fieldName, mlir::Type recTy,
                              mlir::ValueRange operands) {
  result.addAttribute(getFieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(recTy));
  result.addOperands(operands);
}

llvm::SmallVector<mlir::Attribute> fir::FieldIndexOp::getAttributes() {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.push_back(getFieldIdAttr());
  attrs.push_back(getOnTypeAttr());
  return attrs;
}

//===----------------------------------------------------------------------===//
// InsertOnRangeOp
//===----------------------------------------------------------------------===//

static mlir::ParseResult
parseCustomRangeSubscript(mlir::OpAsmParser &parser,
                          mlir::DenseIntElementsAttr &coord) {
  llvm::SmallVector<std::int64_t> lbounds;
  llvm::SmallVector<std::int64_t> ubounds;
  if (parser.parseKeyword("from") ||
      parser.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::Paren,
          [&] { return parser.parseInteger(lbounds.emplace_back(0)); }) ||
      parser.parseKeyword("to") ||
      parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren, [&] {
        return parser.parseInteger(ubounds.emplace_back(0));
      }))
    return mlir::failure();
  llvm::SmallVector<std::int64_t> zippedBounds;
  for (auto zip : llvm::zip(lbounds, ubounds)) {
    zippedBounds.push_back(std::get<0>(zip));
    zippedBounds.push_back(std::get<1>(zip));
  }
  coord = mlir::Builder(parser.getContext()).getIndexTensorAttr(zippedBounds);
  return mlir::success();
}

static void printCustomRangeSubscript(mlir::OpAsmPrinter &printer,
                                      fir::InsertOnRangeOp op,
                                      mlir::DenseIntElementsAttr coord) {
  printer << "from (";
  auto enumerate = llvm::enumerate(coord.getValues<std::int64_t>());
  // Even entries are the lower bounds.
  llvm::interleaveComma(
      make_filter_range(
          enumerate,
          [](auto indexed_value) { return indexed_value.index() % 2 == 0; }),
      printer, [&](auto indexed_value) { printer << indexed_value.value(); });
  printer << ") to (";
  // Odd entries are the upper bounds.
  llvm::interleaveComma(
      make_filter_range(
          enumerate,
          [](auto indexed_value) { return indexed_value.index() % 2 != 0; }),
      printer, [&](auto indexed_value) { printer << indexed_value.value(); });
  printer << ")";
}

/// Range bounds must be nonnegative, and the range must not be empty.
llvm::LogicalResult fir::InsertOnRangeOp::verify() {
  if (fir::hasDynamicSize(getSeq().getType()))
    return emitOpError("must have constant shape and size");
  mlir::DenseIntElementsAttr coorAttr = getCoor();
  if (coorAttr.size() < 2 || coorAttr.size() % 2 != 0)
    return emitOpError("has uneven number of values in ranges");
  bool rangeIsKnownToBeNonempty = false;
  for (auto i = coorAttr.getValues<std::int64_t>().end(),
            b = coorAttr.getValues<std::int64_t>().begin();
       i != b;) {
    int64_t ub = (*--i);
    int64_t lb = (*--i);
    if (lb < 0 || ub < 0)
      return emitOpError("negative range bound");
    if (rangeIsKnownToBeNonempty)
      continue;
    if (lb > ub)
      return emitOpError("empty range");
    rangeIsKnownToBeNonempty = lb < ub;
  }
  return mlir::success();
}

bool fir::InsertOnRangeOp::isFullRange() {
  auto extents = getType().getShape();
  mlir::DenseIntElementsAttr indexes = getCoor();
  if (indexes.size() / 2 != static_cast<int64_t>(extents.size()))
    return false;
  auto cur_index = indexes.value_begin<int64_t>();
  for (unsigned i = 0; i < indexes.size(); i += 2) {
    if (*(cur_index++) != 0)
      return false;
    if (*(cur_index++) != extents[i / 2] - 1)
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

static bool checkIsIntegerConstant(mlir::Attribute attr, std::int64_t conVal) {
  if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
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

  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insval = mlir::dyn_cast_or_null<fir::InsertValueOp>(op);
    if (!insval || !mlir::isa<mlir::ComplexType>(insval.getType()))
      return mlir::failure();
    auto insval2 = mlir::dyn_cast_or_null<fir::InsertValueOp>(
        insval.getAdt().getDefiningOp());
    if (!insval2)
      return mlir::failure();
    auto binf = mlir::dyn_cast_or_null<FltOp>(insval.getVal().getDefiningOp());
    auto binf2 =
        mlir::dyn_cast_or_null<FltOp>(insval2.getVal().getDefiningOp());
    if (!binf || !binf2 || insval.getCoor().size() != 1 ||
        !isOne(insval.getCoor()[0]) || insval2.getCoor().size() != 1 ||
        !isZero(insval2.getCoor()[0]))
      return mlir::failure();
    auto eai = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getLhs().getDefiningOp());
    auto ebi = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getRhs().getDefiningOp());
    auto ear = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getLhs().getDefiningOp());
    auto ebr = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getRhs().getDefiningOp());
    if (!eai || !ebi || !ear || !ebr || ear.getAdt() != eai.getAdt() ||
        ebr.getAdt() != ebi.getAdt() || eai.getCoor().size() != 1 ||
        !isOne(eai.getCoor()[0]) || ebi.getCoor().size() != 1 ||
        !isOne(ebi.getCoor()[0]) || ear.getCoor().size() != 1 ||
        !isZero(ear.getCoor()[0]) || ebr.getCoor().size() != 1 ||
        !isZero(ebr.getCoor()[0]))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<CpxOp>(op, ear.getAdt(), ebr.getAdt());
    return mlir::success();
  }
};

void fir::InsertValueOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<UndoComplexPattern<mlir::arith::AddFOp, fir::AddcOp>,
                 UndoComplexPattern<mlir::arith::SubFOp, fir::SubcOp>>(context);
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
    result.addAttribute(getFinalValueAttrNameStr(), builder.getUnitAttr());
  }
  result.addTypes(iterate.getType());
  result.addOperands(iterArgs);
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArgument(iterate.getType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<mlir::Location>(iterArgs.size(), result.location));
  result.addAttributes(attributes);
}

mlir::ParseResult fir::IterWhileOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::Argument inductionVariable, iterateVar;
  mlir::OpAsmParser::UnresolvedOperand lb, ub, step, iterateInput;
  if (parser.parseLParen() || parser.parseArgument(inductionVariable) ||
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
      parser.resolveOperand(step, indexType, result.operands) ||
      parser.parseKeyword("and") || parser.parseLParen() ||
      parser.parseArgument(iterateVar) || parser.parseEqual() ||
      parser.parseOperand(iterateInput) || parser.parseRParen() ||
      parser.resolveOperand(iterateInput, i1Type, result.operands))
    return mlir::failure();

  // Parse the initial iteration arguments.
  auto prependCount = false;

  // Induction variable.
  llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(iterateVar);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> regionTypes;
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(regionTypes))
      return mlir::failure();
    if (regionTypes.size() == operands.size() + 2)
      prependCount = true;
    llvm::ArrayRef<mlir::Type> resTypes = regionTypes;
    resTypes = prependCount ? resTypes.drop_front(2) : resTypes;
    // Resolve input operands.
    for (auto operandType : llvm::zip(operands, resTypes))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return mlir::failure();
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
      return mlir::failure();
    // Type list must be "(index, i1)".
    if (typeList.size() != 2 || !mlir::isa<mlir::IndexType>(typeList[0]) ||
        !typeList[1].isSignlessInteger(1))
      return mlir::failure();
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
    result.addAttribute(IterWhileOp::getFinalValueAttrNameStr(),
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

  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = argTypes[i];

  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  fir::IterWhileOp::ensureTerminator(*body, builder, result.location);
  return mlir::success();
}

llvm::LogicalResult fir::IterWhileOp::verify() {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (!body->getArgument(1).getType().isInteger(1))
    return emitOpError(
        "expected body second argument to be an index argument for "
        "the induction variable");
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = getNumResults();
  if (getFinalValue()) {
    // Result type must be "(index, i1, ...)".
    if (!mlir::isa<mlir::IndexType>(getResult(0).getType()))
      return emitOpError("result #0 expected to be index");
    if (!getResult(1).getType().isSignlessInteger(1))
      return emitOpError("result #1 expected to be i1");
    opNumResults--;
  } else {
    // iterate_while always returns the early exit induction value.
    // Result type must be "(i1, ...)"
    if (!getResult(0).getType().isSignlessInteger(1))
      return emitOpError("result #0 expected to be i1");
  }
  if (opNumResults == 0)
    return mlir::failure();
  if (getNumIterOperands() != opNumResults)
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (getNumRegionIterArgs() != opNumResults)
    return emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = getIterOperands();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getFinalValue() ? getResults().drop_front() : getResults();
  unsigned i = 0u;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    i++;
  }
  return mlir::success();
}

void fir::IterWhileOp::print(mlir::OpAsmPrinter &p) {
  p << " (" << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep() << ") and (";
  assert(hasIterOperands());
  auto regionArgs = getRegionIterArgs();
  auto operands = getIterOperands();
  p << regionArgs.front() << " = " << *operands.begin() << ")";
  if (regionArgs.size() > 1) {
    p << " iter_args(";
    llvm::interleaveComma(
        llvm::zip(regionArgs.drop_front(), operands.drop_front()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ") -> (";
    llvm::interleaveComma(
        llvm::drop_begin(getResultTypes(), getFinalValue() ? 0 : 1), p);
    p << ")";
  } else if (getFinalValue()) {
    p << " -> (" << getResultTypes() << ')';
  }
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {getFinalValueAttrNameStr()});
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

llvm::SmallVector<mlir::Region *> fir::IterWhileOp::getLoopRegions() {
  return {&getRegion()};
}

mlir::BlockArgument fir::IterWhileOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

void fir::IterWhileOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

mlir::Value fir::IterWhileOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

std::optional<llvm::MutableArrayRef<mlir::OpOperand>>
fir::IterWhileOp::getYieldedValuesMutable() {
  auto *term = getRegion().front().getTerminator();
  return getFinalValue() ? term->getOpOperands().drop_front()
                         : term->getOpOperands();
}

//===----------------------------------------------------------------------===//
// LenParamIndexOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::LenParamIndexOp::parse(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result) {
  return parseFieldLikeOp<fir::LenType>(parser, result);
}

void fir::LenParamIndexOp::print(mlir::OpAsmPrinter &p) {
  printFieldLikeOp(p, *this);
}

void fir::LenParamIndexOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 llvm::StringRef fieldName, mlir::Type recTy,
                                 mlir::ValueRange operands) {
  result.addAttribute(getFieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(recTy));
  result.addOperands(operands);
}

llvm::SmallVector<mlir::Attribute> fir::LenParamIndexOp::getAttributes() {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.push_back(getFieldIdAttr());
  attrs.push_back(getOnTypeAttr());
  return attrs;
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
  build(builder, result, eleTy, refVal);
}

void fir::LoadOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::Type resTy, mlir::Value refVal) {

  if (!refVal) {
    mlir::emitError(result.location, "LoadOp has null argument");
    return;
  }
  result.addOperands(refVal);
  result.addTypes(resTy);
}

mlir::ParseResult fir::LoadOp::getElementOf(mlir::Type &ele, mlir::Type ref) {
  if ((ele = fir::dyn_cast_ptrEleTy(ref)))
    return mlir::success();
  return mlir::failure();
}

mlir::ParseResult fir::LoadOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::UnresolvedOperand oper;
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

void fir::LoadOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getMemref());
  p.printOptionalAttrDict(getOperation()->getAttrs(), {});
  p << " : " << getMemref().getType();
}

void fir::LoadOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getMemrefMutable(),
                       mlir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// DoLoopOp
//===----------------------------------------------------------------------===//

void fir::DoLoopOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Value lb,
                          mlir::Value ub, mlir::Value step, bool unordered,
                          bool finalCountValue, mlir::ValueRange iterArgs,
                          mlir::ValueRange reduceOperands,
                          llvm::ArrayRef<mlir::Attribute> reduceAttrs,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step});
  result.addOperands(reduceOperands);
  result.addOperands(iterArgs);
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, 1, 1, static_cast<int32_t>(reduceOperands.size()),
                           static_cast<int32_t>(iterArgs.size())}));
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(getFinalValueAttrName(result.name),
                        builder.getUnitAttr());
  }
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (iterArgs.empty() && !finalCountValue)
    fir::DoLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<mlir::Location>(iterArgs.size(), result.location));
  if (unordered)
    result.addAttribute(getUnorderedAttrName(result.name),
                        builder.getUnitAttr());
  if (!reduceAttrs.empty())
    result.addAttribute(getReduceAttrsAttrName(result.name),
                        builder.getArrayAttr(reduceAttrs));
  result.addAttributes(attributes);
}

mlir::ParseResult fir::DoLoopOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::Argument inductionVariable;
  mlir::OpAsmParser::UnresolvedOperand lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return mlir::failure();

  if (mlir::succeeded(parser.parseOptionalKeyword("unordered")))
    result.addAttribute("unordered", builder.getUnitAttr());

  // Parse the reduction arguments.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> reduceOperands;
  llvm::SmallVector<mlir::Type> reduceArgTypes;
  if (succeeded(parser.parseOptionalKeyword("reduce"))) {
    // Parse reduction attributes and variables.
    llvm::SmallVector<ReduceAttr> attributes;
    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Paren, [&]() {
              if (parser.parseAttribute(attributes.emplace_back()) ||
                  parser.parseArrow() ||
                  parser.parseOperand(reduceOperands.emplace_back()) ||
                  parser.parseColonType(reduceArgTypes.emplace_back()))
                return mlir::failure();
              return mlir::success();
            })))
      return mlir::failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(reduceOperands, reduceArgTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return mlir::failure();
    llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                                 attributes.end());
    result.addAttribute(getReduceAttrsAttrName(result.name),
                        builder.getArrayAttr(arrayAttr));
  }

  // Parse the optional initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> iterOperands;
  llvm::SmallVector<mlir::Type> argTypes;
  bool prependCount = false;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, iterOperands) ||
        parser.parseArrowTypeList(result.types))
      return mlir::failure();
    if (result.types.size() == iterOperands.size() + 1)
      prependCount = true;
    // Resolve input operands.
    llvm::ArrayRef<mlir::Type> resTypes = result.types;
    for (auto operand_type : llvm::zip(
             iterOperands, prependCount ? resTypes.drop_front() : resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return mlir::failure();
  } else if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseKeyword("index"))
      return mlir::failure();
    result.types.push_back(indexType);
    prependCount = true;
  }

  // Set the operandSegmentSizes attribute
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, 1, 1, static_cast<int32_t>(reduceOperands.size()),
                           static_cast<int32_t>(iterOperands.size())}));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  // Induction variable.
  if (prependCount)
    result.addAttribute(DoLoopOp::getFinalValueAttrName(result.name),
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
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = argTypes[i];

  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  DoLoopOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

fir::DoLoopOp fir::getForInductionVarOwner(mlir::Value val) {
  auto ivArg = mlir::dyn_cast<mlir::BlockArgument>(val);
  if (!ivArg)
    return {};
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return mlir::dyn_cast_or_null<fir::DoLoopOp>(containingInst);
}

// Lifted from loop.loop
llvm::LogicalResult fir::DoLoopOp::verify() {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = getNumResults();
  if (opNumResults == 0)
    return mlir::success();

  if (getFinalValue()) {
    if (getUnordered())
      return emitOpError("unordered loop has no final value");
    opNumResults--;
  }
  if (getNumIterOperands() != opNumResults)
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (getNumRegionIterArgs() != opNumResults)
    return emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = getIterOperands();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getFinalValue() ? getResults().drop_front() : getResults();
  unsigned i = 0u;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    i++;
  }
  auto reduceAttrs = getReduceAttrsAttr();
  if (getNumReduceOperands() != (reduceAttrs ? reduceAttrs.size() : 0))
    return emitOpError(
        "mismatch in number of reduction variables and reduction attributes");
  return mlir::success();
}

void fir::DoLoopOp::print(mlir::OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  p << ' ' << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();
  if (getUnordered())
    p << " unordered";
  if (hasReduceOperands()) {
    p << " reduce(";
    auto attrs = getReduceAttrsAttr();
    auto operands = getReduceOperands();
    llvm::interleaveComma(llvm::zip(attrs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " -> " << std::get<1>(it) << " : "
        << std::get<1>(it).getType();
    });
    p << ')';
    printBlockTerminators = true;
  }
  if (hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = getRegionIterArgs();
    auto operands = getIterOperands();
    llvm::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ") -> (" << getResultTypes() << ')';
    printBlockTerminators = true;
  } else if (getFinalValue()) {
    p << " -> " << getResultTypes();
    printBlockTerminators = true;
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {"unordered", "finalValue", "reduceAttrs", "operandSegmentSizes"});
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
}

llvm::SmallVector<mlir::Region *> fir::DoLoopOp::getLoopRegions() {
  return {&getRegion()};
}

/// Translate a value passed as an iter_arg to the corresponding block
/// argument in the body of the loop.
mlir::BlockArgument fir::DoLoopOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

/// Translate the result vector (by index number) to the corresponding value
/// to the `fir.result` Op.
void fir::DoLoopOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

/// Translate the block argument (by index number) to the corresponding value
/// passed as an iter_arg to the parent DoLoopOp.
mlir::Value fir::DoLoopOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

std::optional<llvm::MutableArrayRef<mlir::OpOperand>>
fir::DoLoopOp::getYieldedValuesMutable() {
  auto *term = getRegion().front().getTerminator();
  return getFinalValue() ? term->getOpOperands().drop_front()
                         : term->getOpOperands();
}

//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::DTEntryOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  llvm::StringRef methodName;
  // allow `methodName` or `"methodName"`
  if (failed(parser.parseOptionalKeyword(&methodName))) {
    mlir::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr, getMethodAttrName(result.name),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(getMethodAttrName(result.name),
                        parser.getBuilder().getStringAttr(methodName));
  }
  mlir::SymbolRefAttr calleeAttr;
  if (parser.parseComma() ||
      parser.parseAttribute(calleeAttr, fir::DTEntryOp::getProcAttrNameStr(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

void fir::DTEntryOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getMethodAttr() << ", " << getProcAttr();
}

//===----------------------------------------------------------------------===//
// ReboxOp
//===----------------------------------------------------------------------===//

/// Get the scalar type related to a fir.box type.
/// Example: return f32 for !fir.box<!fir.heap<!fir.array<?x?xf32>>.
static mlir::Type getBoxScalarEleTy(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(eleTy))
    return seqTy.getEleTy();
  return eleTy;
}

/// Test if \p t1 and \p t2 are compatible character types (if they can
/// represent the same type at runtime).
static bool areCompatibleCharacterTypes(mlir::Type t1, mlir::Type t2) {
  auto c1 = mlir::dyn_cast<fir::CharacterType>(t1);
  auto c2 = mlir::dyn_cast<fir::CharacterType>(t2);
  if (!c1 || !c2)
    return false;
  if (c1.hasDynamicLen() || c2.hasDynamicLen())
    return true;
  return c1.getLen() == c2.getLen();
}

llvm::LogicalResult fir::ReboxOp::verify() {
  auto inputBoxTy = getBox().getType();
  if (fir::isa_unknown_size_box(inputBoxTy))
    return emitOpError("box operand must not have unknown rank or type");
  auto outBoxTy = getType();
  if (fir::isa_unknown_size_box(outBoxTy))
    return emitOpError("result type must not have unknown rank or type");
  auto inputRank = fir::getBoxRank(inputBoxTy);
  auto inputEleTy = getBoxScalarEleTy(inputBoxTy);
  auto outRank = fir::getBoxRank(outBoxTy);
  auto outEleTy = getBoxScalarEleTy(outBoxTy);

  if (auto sliceVal = getSlice()) {
    // Slicing case
    if (mlir::cast<fir::SliceType>(sliceVal.getType()).getRank() != inputRank)
      return emitOpError("slice operand rank must match box operand rank");
    if (auto shapeVal = getShape()) {
      if (auto shiftTy = mlir::dyn_cast<fir::ShiftType>(shapeVal.getType())) {
        if (shiftTy.getRank() != inputRank)
          return emitOpError("shape operand and input box ranks must match "
                             "when there is a slice");
      } else {
        return emitOpError("shape operand must absent or be a fir.shift "
                           "when there is a slice");
      }
    }
    if (auto sliceOp = sliceVal.getDefiningOp()) {
      auto slicedRank = mlir::cast<fir::SliceOp>(sliceOp).getOutRank();
      if (slicedRank != outRank)
        return emitOpError("result type rank and rank after applying slice "
                           "operand must match");
    }
  } else {
    // Reshaping case
    unsigned shapeRank = inputRank;
    if (auto shapeVal = getShape()) {
      auto ty = shapeVal.getType();
      if (auto shapeTy = mlir::dyn_cast<fir::ShapeType>(ty)) {
        shapeRank = shapeTy.getRank();
      } else if (auto shapeShiftTy = mlir::dyn_cast<fir::ShapeShiftType>(ty)) {
        shapeRank = shapeShiftTy.getRank();
      } else {
        auto shiftTy = mlir::cast<fir::ShiftType>(ty);
        shapeRank = shiftTy.getRank();
        if (shapeRank != inputRank)
          return emitOpError("shape operand and input box ranks must match "
                             "when the shape is a fir.shift");
      }
    }
    if (shapeRank != outRank)
      return emitOpError("result type and shape operand ranks must match");
  }

  if (inputEleTy != outEleTy) {
    // TODO: check that outBoxTy is a parent type of inputBoxTy for derived
    // types.
    // Character input and output types with constant length may be different if
    // there is a substring in the slice, otherwise, they must match. If any of
    // the types is a character with dynamic length, the other type can be any
    // character type.
    const bool typeCanMismatch =
        mlir::isa<fir::RecordType>(inputEleTy) ||
        mlir::isa<mlir::NoneType>(outEleTy) ||
        (mlir::isa<mlir::NoneType>(inputEleTy) &&
         mlir::isa<fir::RecordType>(outEleTy)) ||
        (getSlice() && mlir::isa<fir::CharacterType>(inputEleTy)) ||
        (getSlice() && fir::isa_complex(inputEleTy) &&
         mlir::isa<mlir::FloatType>(outEleTy)) ||
        areCompatibleCharacterTypes(inputEleTy, outEleTy);
    if (!typeCanMismatch)
      return emitOpError(
          "op input and output element types must match for intrinsic types");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReboxAssumedRankOp
//===----------------------------------------------------------------------===//

static bool areCompatibleAssumedRankElementType(mlir::Type inputEleTy,
                                                mlir::Type outEleTy) {
  if (inputEleTy == outEleTy)
    return true;
  // Output is unlimited polymorphic -> output dynamic type is the same as input
  // type.
  if (mlir::isa<mlir::NoneType>(outEleTy))
    return true;
  // Output/Input are derived types. Assuming input extends output type, output
  // dynamic type is the output static type, unless output is polymorphic.
  if (mlir::isa<fir::RecordType>(inputEleTy) &&
      mlir::isa<fir::RecordType>(outEleTy))
    return true;
  if (areCompatibleCharacterTypes(inputEleTy, outEleTy))
    return true;
  return false;
}

llvm::LogicalResult fir::ReboxAssumedRankOp::verify() {
  mlir::Type inputType = getBox().getType();
  if (!mlir::isa<fir::BaseBoxType>(inputType) && !fir::isBoxAddress(inputType))
    return emitOpError("input must be a box or box address");
  mlir::Type inputEleTy =
      mlir::cast<fir::BaseBoxType>(fir::unwrapRefType(inputType))
          .unwrapInnerType();
  mlir::Type outEleTy =
      mlir::cast<fir::BaseBoxType>(getType()).unwrapInnerType();
  if (!areCompatibleAssumedRankElementType(inputEleTy, outEleTy))
    return emitOpError("input and output element types are incompatible");
  return mlir::success();
}

void fir::ReboxAssumedRankOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  mlir::OpOperand &inputBox = getBoxMutable();
  if (fir::isBoxAddress(inputBox.get().getType()))
    effects.emplace_back(mlir::MemoryEffects::Read::get(), &inputBox,
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ResultOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ResultOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  auto results = parentOp->getResults();
  auto operands = (*this)->getOperands();

  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of result must have same arity";
  for (auto e : llvm::zip(results, operands))
    if (std::get<0>(e).getType() != std::get<1>(e).getType())
      return emitOpError() << "types mismatch between result op and its parent";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SaveResultOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::SaveResultOp::verify() {
  auto resultType = getValue().getType();
  if (resultType != fir::dyn_cast_ptrEleTy(getMemref().getType()))
    return emitOpError("value type must match memory reference type");
  if (fir::isa_unknown_size_box(resultType))
    return emitOpError("cannot save !fir.box of unknown rank or type");

  if (mlir::isa<fir::BoxType>(resultType)) {
    if (getShape() || !getTypeparams().empty())
      return emitOpError(
          "must not have shape or length operands if the value is a fir.box");
    return mlir::success();
  }

  // fir.record or fir.array case.
  unsigned shapeTyRank = 0;
  if (auto shapeVal = getShape()) {
    auto shapeTy = shapeVal.getType();
    if (auto s = mlir::dyn_cast<fir::ShapeType>(shapeTy))
      shapeTyRank = s.getRank();
    else
      shapeTyRank = mlir::cast<fir::ShapeShiftType>(shapeTy).getRank();
  }

  auto eleTy = resultType;
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(resultType)) {
    if (seqTy.getDimension() != shapeTyRank)
      emitOpError("shape operand must be provided and have the value rank "
                  "when the value is a fir.array");
    eleTy = seqTy.getEleTy();
  } else {
    if (shapeTyRank != 0)
      emitOpError(
          "shape operand should only be provided if the value is a fir.array");
  }

  if (auto recTy = mlir::dyn_cast<fir::RecordType>(eleTy)) {
    if (recTy.getNumLenParams() != getTypeparams().size())
      emitOpError("length parameters number must match with the value type "
                  "length parameters");
  } else if (auto charTy = mlir::dyn_cast<fir::CharacterType>(eleTy)) {
    if (getTypeparams().size() > 1)
      emitOpError("no more than one length parameter must be provided for "
                  "character value");
  } else {
    if (!getTypeparams().empty())
      emitOpError("length parameters must not be provided for this value type");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IntegralSwitchTerminator
//===----------------------------------------------------------------------===//
static constexpr llvm::StringRef getCompareOffsetAttr() {
  return "compare_operand_offsets";
}

static constexpr llvm::StringRef getTargetOffsetAttr() {
  return "target_operand_offsets";
}

template <typename OpT>
static llvm::LogicalResult verifyIntegralSwitchTerminator(OpT op) {
  if (!mlir::isa<mlir::IntegerType, mlir::IndexType, fir::IntegerType>(
          op.getSelector().getType()))
    return op.emitOpError("must be an integer");
  auto cases =
      op->template getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumDest();
  if (count == 0)
    return op.emitOpError("must have at least one successor");
  if (op.getNumConditions() != count)
    return op.emitOpError("number of cases and targets don't match");
  if (op.targetOffsetSize() != count)
    return op.emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    if (!mlir::isa<mlir::IntegerAttr, mlir::UnitAttr>(cases[i]))
      return op.emitOpError("invalid case alternative");
  }
  return mlir::success();
}

static mlir::ParseResult parseIntegralSwitchTerminator(
    mlir::OpAsmParser &parser, mlir::OperationState &result,
    llvm::StringRef casesAttr, llvm::StringRef operandSegmentAttr) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> ivalues;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  while (true) {
    mlir::Attribute ivalue; // Integer or Unit
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(ivalue, "i", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    ivalues.push_back(ivalue);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (!parser.parseOptionalRSquare())
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(casesAttr, bld.getArrayAttr(ivalues));
  llvm::SmallVector<int32_t> argOffs;
  int32_t sumArgs = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    argOffs.push_back(argSize);
    sumArgs += argSize;
  }
  result.addAttribute(operandSegmentAttr,
                      bld.getDenseI32ArrayAttr({1, 0, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), bld.getDenseI32ArrayAttr(argOffs));
  return mlir::success();
}

template <typename OpT>
static void printIntegralSwitchTerminator(OpT op, mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(op.getSelector());
  p << " : " << op.getSelector().getType() << " [";
  auto cases =
      op->template getAttrOfType<mlir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    auto &attr = cases[i];
    if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr))
      p << intAttr.getValue();
    else
      p.printAttribute(attr);
    p << ", ";
    op.printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(
      op->getAttrs(), {op.getCasesAttr(), getCompareOffsetAttr(),
                       getTargetOffsetAttr(), op.getOperandSegmentSizeAttr()});
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::SelectOp::verify() {
  return verifyIntegralSwitchTerminator(*this);
}

mlir::ParseResult fir::SelectOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectOp::print(mlir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

template <typename A, typename... AdditionalArgs>
static A getSubOperands(unsigned pos, A allArgs, mlir::DenseI32ArrayAttr ranges,
                        AdditionalArgs &&...additionalArgs) {
  unsigned start = 0;
  for (unsigned i = 0; i < pos; ++i)
    start += ranges[i];
  return allArgs.slice(start, ranges[pos],
                       std::forward<AdditionalArgs>(additionalArgs)...);
}

static mlir::MutableOperandRange
getMutableSuccessorOperands(unsigned pos, mlir::MutableOperandRange operands,
                            llvm::StringRef offsetAttr) {
  mlir::Operation *owner = operands.getOwner();
  mlir::NamedAttribute targetOffsetAttr =
      *owner->getAttrDictionary().getNamed(offsetAttr);
  return getSubOperands(
      pos, operands,
      mlir::cast<mlir::DenseI32ArrayAttr>(targetOffsetAttr.getValue()),
      mlir::MutableOperandRange::OperandSegment(pos, targetOffsetAttr));
}

std::optional<mlir::OperandRange> fir::SelectOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                    unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<mlir::ValueRange>
fir::SelectOp::getSuccessorOperands(mlir::ValueRange operands, unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

//===----------------------------------------------------------------------===//
// SelectCaseOp
//===----------------------------------------------------------------------===//

std::optional<mlir::OperandRange>
fir::SelectCaseOp::getCompareOperands(unsigned cond) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  return {getSubOperands(cond, getCompareArgs(), a)};
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getCompareOperands(llvm::ArrayRef<mlir::Value> operands,
                                      unsigned cond) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

std::optional<mlir::ValueRange>
fir::SelectCaseOp::getCompareOperands(mlir::ValueRange operands,
                                      unsigned cond) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

mlir::SuccessorOperands fir::SelectCaseOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<mlir::ValueRange>
fir::SelectCaseOp::getSuccessorOperands(mlir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

// parser for fir.select_case Op
mlir::ParseResult fir::SelectCaseOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> attrs;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> opers;
  llvm::SmallVector<mlir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> destArgs;
  llvm::SmallVector<std::int32_t> argOffs;
  std::int32_t offSize = 0;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || isValidCaseAttr(attr) ||
        parser.parseComma())
      return mlir::failure();
    attrs.push_back(attr);
    if (mlir::dyn_cast_or_null<mlir::UnitAttr>(attr)) {
      argOffs.push_back(0);
    } else if (mlir::dyn_cast_or_null<fir::ClosedIntervalAttr>(attr)) {
      mlir::OpAsmParser::UnresolvedOperand oper1;
      mlir::OpAsmParser::UnresolvedOperand oper2;
      if (parser.parseOperand(oper1) || parser.parseComma() ||
          parser.parseOperand(oper2) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper1);
      opers.push_back(oper2);
      argOffs.push_back(2);
      offSize += 2;
    } else {
      mlir::OpAsmParser::UnresolvedOperand oper;
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
                      bld.getDenseI32ArrayAttr({1, offSize, toffSize}));
  result.addAttribute(getCompareOffsetAttr(),
                      bld.getDenseI32ArrayAttr(argOffs));
  result.addAttribute(getTargetOffsetAttr(),
                      bld.getDenseI32ArrayAttr(targOffs));
  return mlir::success();
}

void fir::SelectCaseOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    if (!mlir::isa<mlir::UnitAttr>(cases[i])) {
      auto caseArgs = *getCompareOperands(i);
      p.printOperand(*caseArgs.begin());
      p << ", ";
      if (mlir::isa<fir::ClosedIntervalAttr>(cases[i])) {
        p.printOperand(*(++caseArgs.begin()));
        p << ", ";
      }
    }
    printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(), getOperandSegmentSizeAttr()});
}

unsigned fir::SelectCaseOp::compareOffsetSize() {
  return (*this)
      ->getAttrOfType<mlir::DenseI32ArrayAttr>(getCompareOffsetAttr())
      .size();
}

unsigned fir::SelectCaseOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
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
    if (mlir::isa<fir::ClosedIntervalAttr>(attr)) {
      operOffs.push_back(2);
      operSize += 2;
    } else if (mlir::isa<mlir::UnitAttr>(attr)) {
      operOffs.push_back(0);
    } else {
      operOffs.push_back(1);
      ++operSize;
    }
  }
  for (auto ops : cmpOperands)
    result.addOperands(ops);
  result.addAttribute(getCompareOffsetAttr(),
                      builder.getDenseI32ArrayAttr(operOffs));
  const auto count = destinations.size();
  for (auto d : destinations)
    result.addSuccessors(d);
  const auto opCount = destOperands.size();
  llvm::SmallVector<std::int32_t> argOffs;
  std::int32_t sumArgs = 0;
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
                      builder.getDenseI32ArrayAttr({1, operSize, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(),
                      builder.getDenseI32ArrayAttr(argOffs));
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
    if (mlir::isa<fir::ClosedIntervalAttr>(attr)) {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 2}));
      iter += 2;
    } else if (mlir::isa<mlir::UnitAttr>(attr)) {
      cmpOpers.push_back(mlir::ValueRange{});
    } else {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 1}));
      ++iter;
    }
  }
  build(builder, result, selector, compareAttrs, cmpOpers, destinations,
        destOperands, attributes);
}

llvm::LogicalResult fir::SelectCaseOp::verify() {
  if (!mlir::isa<mlir::IntegerType, mlir::IndexType, fir::IntegerType,
                 fir::LogicalType, fir::CharacterType>(getSelector().getType()))
    return emitOpError("must be an integer, character, or logical");
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumDest();
  if (count == 0)
    return emitOpError("must have at least one successor");
  if (getNumConditions() != count)
    return emitOpError("number of conditions and successors don't match");
  if (compareOffsetSize() != count)
    return emitOpError("incorrect number of compare operand groups");
  if (targetOffsetSize() != count)
    return emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    auto &attr = cases[i];
    if (!(mlir::isa<fir::PointIntervalAttr>(attr) ||
          mlir::isa<fir::LowerBoundAttr>(attr) ||
          mlir::isa<fir::UpperBoundAttr>(attr) ||
          mlir::isa<fir::ClosedIntervalAttr>(attr) ||
          mlir::isa<mlir::UnitAttr>(attr)))
      return emitOpError("incorrect select case attribute type");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SelectRankOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::SelectRankOp::verify() {
  return verifyIntegralSwitchTerminator(*this);
}

mlir::ParseResult fir::SelectRankOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectRankOp::print(mlir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

std::optional<mlir::OperandRange>
fir::SelectRankOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectRankOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<mlir::ValueRange>
fir::SelectRankOp::getSuccessorOperands(mlir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectRankOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

//===----------------------------------------------------------------------===//
// SelectTypeOp
//===----------------------------------------------------------------------===//

std::optional<mlir::OperandRange>
fir::SelectTypeOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

mlir::SuccessorOperands fir::SelectTypeOp::getSuccessorOperands(unsigned oper) {
  return mlir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<mlir::ValueRange>
fir::SelectTypeOp::getSuccessorOperands(mlir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

mlir::ParseResult fir::SelectTypeOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand selector;
  mlir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
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
                      bld.getDenseI32ArrayAttr({1, 0, offSize}));
  result.addAttribute(getTargetOffsetAttr(), bld.getDenseI32ArrayAttr(argOffs));
  return mlir::success();
}

unsigned fir::SelectTypeOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<mlir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

void fir::SelectTypeOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    printSuccessorAtIndex(p, i);
  }
  p << ']';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {getCasesAttr(), getCompareOffsetAttr(),
                           getTargetOffsetAttr(),
                           fir::SelectTypeOp::getOperandSegmentSizeAttr()});
}

llvm::LogicalResult fir::SelectTypeOp::verify() {
  if (!mlir::isa<fir::BaseBoxType>(getSelector().getType()))
    return emitOpError("must be a fir.class or fir.box type");
  if (auto boxType = mlir::dyn_cast<fir::BoxType>(getSelector().getType()))
    if (!mlir::isa<mlir::NoneType>(boxType.getEleTy()))
      return emitOpError("selector must be polymorphic");
  auto typeGuardAttr = getCases();
  for (unsigned idx = 0; idx < typeGuardAttr.size(); ++idx)
    if (mlir::isa<mlir::UnitAttr>(typeGuardAttr[idx]) &&
        idx != typeGuardAttr.size() - 1)
      return emitOpError("default must be the last attribute");
  auto count = getNumDest();
  if (count == 0)
    return emitOpError("must have at least one successor");
  if (getNumConditions() != count)
    return emitOpError("number of conditions and successors don't match");
  if (targetOffsetSize() != count)
    return emitOpError("incorrect number of successor operand groups");
  for (unsigned i = 0; i != count; ++i) {
    if (!mlir::isa<fir::ExactTypeAttr, fir::SubclassAttr, mlir::UnitAttr>(
            typeGuardAttr[i]))
      return emitOpError("invalid type-case alternative");
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
                      builder.getDenseI32ArrayAttr({1, 0, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(),
                      builder.getDenseI32ArrayAttr(argOffs));
  result.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ShapeOp::verify() {
  auto size = getExtents().size();
  auto shapeTy = mlir::dyn_cast<fir::ShapeType>(getType());
  assert(shapeTy && "must be a shape type");
  if (shapeTy.getRank() != size)
    return emitOpError("shape type rank mismatch");
  return mlir::success();
}

void fir::ShapeOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::ValueRange extents) {
  auto type = fir::ShapeType::get(builder.getContext(), extents.size());
  build(builder, result, type, extents);
}

//===----------------------------------------------------------------------===//
// ShapeShiftOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ShapeShiftOp::verify() {
  auto size = getPairs().size();
  if (size < 2 || size > 16 * 2)
    return emitOpError("incorrect number of args");
  if (size % 2 != 0)
    return emitOpError("requires a multiple of 2 args");
  auto shapeTy = mlir::dyn_cast<fir::ShapeShiftType>(getType());
  assert(shapeTy && "must be a shape shift type");
  if (shapeTy.getRank() * 2 != size)
    return emitOpError("shape type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ShiftOp::verify() {
  auto size = getOrigins().size();
  auto shiftTy = mlir::dyn_cast<fir::ShiftType>(getType());
  assert(shiftTy && "must be a shift type");
  if (shiftTy.getRank() != size)
    return emitOpError("shift type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void fir::SliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::ValueRange trips, mlir::ValueRange path,
                         mlir::ValueRange substr) {
  const auto rank = trips.size() / 3;
  auto sliceTy = fir::SliceType::get(builder.getContext(), rank);
  build(builder, result, sliceTy, trips, path, substr);
}

/// Return the output rank of a slice op. The output rank must be between 1 and
/// the rank of the array being sliced (inclusive).
unsigned fir::SliceOp::getOutputRank(mlir::ValueRange triples) {
  unsigned rank = 0;
  if (!triples.empty()) {
    for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
      auto *op = triples[i].getDefiningOp();
      if (!mlir::isa_and_nonnull<fir::UndefOp>(op))
        ++rank;
    }
    assert(rank > 0);
  }
  return rank;
}

llvm::LogicalResult fir::SliceOp::verify() {
  auto size = getTriples().size();
  if (size < 3 || size > 16 * 3)
    return emitOpError("incorrect number of args for triple");
  if (size % 3 != 0)
    return emitOpError("requires a multiple of 3 args");
  auto sliceTy = mlir::dyn_cast<fir::SliceType>(getType());
  assert(sliceTy && "must be a slice type");
  if (sliceTy.getRank() * 3 != size)
    return emitOpError("slice type rank mismatch");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

mlir::Type fir::StoreOp::elementType(mlir::Type refType) {
  return fir::dyn_cast_ptrEleTy(refType);
}

mlir::ParseResult fir::StoreOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::Type type;
  mlir::OpAsmParser::UnresolvedOperand oper;
  mlir::OpAsmParser::UnresolvedOperand store;
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

void fir::StoreOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getValue());
  p << " to ";
  p.printOperand(getMemref());
  p.printOptionalAttrDict(getOperation()->getAttrs(), {});
  p << " : " << getMemref().getType();
}

llvm::LogicalResult fir::StoreOp::verify() {
  if (getValue().getType() != fir::dyn_cast_ptrEleTy(getMemref().getType()))
    return emitOpError("store value type must match memory reference type");
  return mlir::success();
}

void fir::StoreOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::Value value, mlir::Value memref) {
  build(builder, result, value, memref, {});
}

void fir::StoreOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getMemrefMutable(),
                       mlir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

void fir::CopyOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::Value source, mlir::Value destination,
                        bool noOverlap) {
  mlir::UnitAttr noOverlapAttr =
      noOverlap ? builder.getUnitAttr() : mlir::UnitAttr{};
  build(builder, result, source, destination, noOverlapAttr);
}

llvm::LogicalResult fir::CopyOp::verify() {
  mlir::Type sourceType = fir::unwrapRefType(getSource().getType());
  mlir::Type destinationType = fir::unwrapRefType(getDestination().getType());
  if (sourceType != destinationType)
    return emitOpError("source and destination must have the same value type");
  return mlir::success();
}

void fir::CopyOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSourceMutable(),
                       mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getDestinationMutable(),
                       mlir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getDestination().getType(), getSource().getType()},
                           effects);
}

//===----------------------------------------------------------------------===//
// StringLitOp
//===----------------------------------------------------------------------===//

inline fir::CharacterType::KindTy stringLitOpGetKind(fir::StringLitOp op) {
  auto eleTy = mlir::cast<fir::SequenceType>(op.getType()).getElementType();
  return mlir::cast<fir::CharacterType>(eleTy).getFKind();
}

bool fir::StringLitOp::isWideValue() { return stringLitOpGetKind(*this) != 1; }

static mlir::NamedAttribute
mkNamedIntegerAttr(mlir::OpBuilder &builder, llvm::StringRef name, int64_t v) {
  assert(v > 0);
  return builder.getNamedAttr(
      name, builder.getIntegerAttr(builder.getIntegerType(64), v));
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType, llvm::StringRef val,
                             std::optional<int64_t> len) {
  auto valAttr = builder.getNamedAttr(value(), builder.getStringAttr(val));
  int64_t length = len ? *len : inType.getLen();
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

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char> vlist,
                             std::optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len ? *len : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char16_t> vlist,
                             std::optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len ? *len : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

void fir::StringLitOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             fir::CharacterType inType,
                             llvm::ArrayRef<char32_t> vlist,
                             std::optional<std::int64_t> len) {
  auto valAttr =
      builder.getNamedAttr(xlist(), convertToArrayAttr(builder, vlist));
  std::int64_t length = len ? *len : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

mlir::ParseResult fir::StringLitOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::Attribute val;
  mlir::NamedAttrList attrs;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseAttribute(val, "fake", attrs))
    return mlir::failure();
  if (auto v = mlir::dyn_cast<mlir::StringAttr>(val))
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::value(), v));
  else if (auto v = mlir::dyn_cast<mlir::DenseElementsAttr>(val))
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::xlist(), v));
  else if (auto v = mlir::dyn_cast<mlir::ArrayAttr>(val))
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
  auto charTy = mlir::dyn_cast<fir::CharacterType>(type);
  if (!charTy)
    return parser.emitError(trailingTypeLoc, "must have character type");
  type = fir::CharacterType::get(builder.getContext(), charTy.getFKind(),
                                 sz.getInt());
  if (!type || parser.addTypesToList(type, result.types))
    return mlir::failure();
  return mlir::success();
}

void fir::StringLitOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getValue() << '(';
  p << mlir::cast<mlir::IntegerAttr>(getSize()).getValue() << ") : ";
  p.printType(getType());
}

llvm::LogicalResult fir::StringLitOp::verify() {
  if (mlir::cast<mlir::IntegerAttr>(getSize()).getValue().isNegative())
    return emitOpError("size must be non-negative");
  if (auto xl = getOperation()->getAttr(fir::StringLitOp::xlist())) {
    if (auto xList = mlir::dyn_cast<mlir::ArrayAttr>(xl)) {
      for (auto a : xList)
        if (!mlir::isa<mlir::IntegerAttr>(a))
          return emitOpError("values in initializer must be integers");
    } else if (mlir::isa<mlir::DenseElementsAttr>(xl)) {
      // do nothing
    } else {
      return emitOpError("has unexpected attribute");
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnboxProcOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::UnboxProcOp::verify() {
  if (auto eleTy = fir::dyn_cast_ptrEleTy(getRefTuple().getType()))
    if (mlir::isa<mlir::TupleType>(eleTy))
      return mlir::success();
  return emitOpError("second output argument has bad type");
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void fir::IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::Value cond, bool withElseRegion) {
  build(builder, result, std::nullopt, cond, withElseRegion);
}

void fir::IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
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

// These 3 functions copied from scf.if implementation.

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control.
void fir::IfOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(mlir::RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  regions.push_back(mlir::RegionSuccessor(&getThenRegion()));

  // Don't consider the else region if it is empty.
  mlir::Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    regions.push_back(mlir::RegionSuccessor());
  else
    regions.push_back(mlir::RegionSuccessor(elseRegion));
}

void fir::IfOp::getEntrySuccessorRegions(
    llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands);
  auto boolAttr =
      mlir::dyn_cast_or_null<mlir::BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&getThenRegion());

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    else
      regions.emplace_back(getResults());
  }
}

void fir::IfOp::getRegionInvocationBounds(
    llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds) {
  if (auto cond = mlir::dyn_cast_or_null<mlir::BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

mlir::ParseResult fir::IfOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::UnresolvedOperand cond;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return mlir::failure();
  fir::IfOp::ensureTerminator(*thenRegion, parser.getBuilder(),
                              result.location);

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return mlir::failure();
    fir::IfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                                result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  return mlir::success();
}

llvm::LogicalResult fir::IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");

  return mlir::success();
}

void fir::IfOp::print(mlir::OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  p << ' ' << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ')';
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &otherReg = getElseRegion();
  if (!otherReg.empty()) {
    p << " else ";
    p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
                  printBlockTerminators);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

void fir::IfOp::resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                                  unsigned resultNum) {
  auto *term = getThenRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
  term = getElseRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
}

//===----------------------------------------------------------------------===//
// BoxOffsetOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::BoxOffsetOp::verify() {
  auto boxType = mlir::dyn_cast_or_null<fir::BaseBoxType>(
      fir::dyn_cast_ptrEleTy(getBoxRef().getType()));
  if (!boxType)
    return emitOpError("box_ref operand must have !fir.ref<!fir.box<T>> type");
  if (getField() != fir::BoxFieldAttr::base_addr &&
      getField() != fir::BoxFieldAttr::derived_type)
    return emitOpError("cannot address provided field");
  if (getField() == fir::BoxFieldAttr::derived_type)
    if (!fir::boxHasAddendum(boxType))
      return emitOpError("can only address derived_type field of derived type "
                         "or unlimited polymorphic fir.box");
  return mlir::success();
}

void fir::BoxOffsetOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value boxRef,
                             fir::BoxFieldAttr field) {
  mlir::Type valueType =
      fir::unwrapPassByRefType(fir::unwrapRefType(boxRef.getType()));
  mlir::Type resultType = valueType;
  if (field == fir::BoxFieldAttr::base_addr)
    resultType = fir::LLVMPointerType::get(fir::ReferenceType::get(valueType));
  else if (field == fir::BoxFieldAttr::derived_type)
    resultType = fir::LLVMPointerType::get(
        fir::TypeDescType::get(fir::unwrapSequenceType(valueType)));
  build(builder, result, {resultType}, boxRef, field);
}

//===----------------------------------------------------------------------===//

mlir::ParseResult fir::isValidCaseAttr(mlir::Attribute attr) {
  if (mlir::isa<mlir::UnitAttr, fir::ClosedIntervalAttr, fir::PointIntervalAttr,
                fir::LowerBoundAttr, fir::UpperBoundAttr>(attr))
    return mlir::success();
  return mlir::failure();
}

unsigned fir::getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                                    unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!mlir::dyn_cast_or_null<mlir::UnitAttr>(attr)) {
      ++o;
      if (mlir::dyn_cast_or_null<fir::ClosedIntervalAttr>(attr))
        ++o;
    }
  }
  return o;
}

mlir::ParseResult
fir::parseSelector(mlir::OpAsmParser &parser, mlir::OperationState &result,
                   mlir::OpAsmParser::UnresolvedOperand &selector,
                   mlir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return mlir::failure();
  return mlir::success();
}

mlir::func::FuncOp fir::createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType type,
                                     llvm::ArrayRef<mlir::NamedAttribute> attrs,
                                     const mlir::SymbolTable *symbolTable) {
  if (symbolTable)
    if (auto f = symbolTable->lookup<mlir::func::FuncOp>(name)) {
#ifdef EXPENSIVE_CHECKS
      assert(f == module.lookupSymbol<mlir::func::FuncOp>(name) &&
             "symbolTable and module out of sync");
#endif
      return f;
    }
  if (auto f = module.lookupSymbol<mlir::func::FuncOp>(name))
    return f;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPointToEnd(module.getBody());
  auto result = modBuilder.create<mlir::func::FuncOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

fir::GlobalOp fir::createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                                  llvm::StringRef name, mlir::Type type,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs,
                                  const mlir::SymbolTable *symbolTable) {
  if (symbolTable)
    if (auto g = symbolTable->lookup<fir::GlobalOp>(name)) {
#ifdef EXPENSIVE_CHECKS
      assert(g == module.lookupSymbol<fir::GlobalOp>(name) &&
             "symbolTable and module out of sync");
#endif
      return g;
    }
  if (auto g = module.lookupSymbol<fir::GlobalOp>(name))
    return g;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  auto result = modBuilder.create<fir::GlobalOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

bool fir::hasHostAssociationArgument(mlir::func::FuncOp func) {
  if (auto allArgAttrs = func.getAllArgAttrs())
    for (auto attr : allArgAttrs)
      if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr))
        if (dict.get(fir::getHostAssocAttrName()))
          return true;
  return false;
}

// Test if value's definition has the specified set of
// attributeNames. The value's definition is one of the operations
// that are able to carry the Fortran variable attributes, e.g.
// fir.alloca or fir.allocmem. Function arguments may also represent
// value definitions and carry relevant attributes.
//
// If it is not possible to reach the limited set of definition
// entities from the given value, then the function will return
// std::nullopt. Otherwise, the definition is known and the return
// value is computed as:
//   * if checkAny is true, then the function will return true
//     iff any of the attributeNames attributes is set on the definition.
//   * if checkAny is false, then the function will return true
//     iff all of the attributeNames attributes are set on the definition.
static std::optional<bool>
valueCheckFirAttributes(mlir::Value value,
                        llvm::ArrayRef<llvm::StringRef> attributeNames,
                        bool checkAny) {
  auto testAttributeSets = [&](llvm::ArrayRef<mlir::NamedAttribute> setAttrs,
                               llvm::ArrayRef<llvm::StringRef> checkAttrs) {
    if (checkAny) {
      // Return true iff any of checkAttrs attributes is present
      // in setAttrs set.
      for (llvm::StringRef checkAttrName : checkAttrs)
        if (llvm::any_of(setAttrs, [&](mlir::NamedAttribute setAttr) {
              return setAttr.getName() == checkAttrName;
            }))
          return true;

      return false;
    }

    // Return true iff all attributes from checkAttrs are present
    // in setAttrs set.
    for (mlir::StringRef checkAttrName : checkAttrs)
      if (llvm::none_of(setAttrs, [&](mlir::NamedAttribute setAttr) {
            return setAttr.getName() == checkAttrName;
          }))
        return false;

    return true;
  };
  // If this is a fir.box that was loaded, the fir attributes will be on the
  // related fir.ref<fir.box> creation.
  if (mlir::isa<fir::BoxType>(value.getType()))
    if (auto definingOp = value.getDefiningOp())
      if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(definingOp))
        value = loadOp.getMemref();
  // If this is a function argument, look in the argument attributes.
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock())
      if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(
              blockArg.getOwner()->getParentOp()))
        return testAttributeSets(
            mlir::cast<mlir::FunctionOpInterface>(*funcOp).getArgAttrs(
                blockArg.getArgNumber()),
            attributeNames);

    // If it is not a function argument, the attributes are unknown.
    return std::nullopt;
  }

  if (auto definingOp = value.getDefiningOp()) {
    // If this is an allocated value, look at the allocation attributes.
    if (mlir::isa<fir::AllocMemOp>(definingOp) ||
        mlir::isa<fir::AllocaOp>(definingOp))
      return testAttributeSets(definingOp->getAttrs(), attributeNames);
    // If this is an imported global, look at AddrOfOp and GlobalOp attributes.
    // Both operations are looked at because use/host associated variable (the
    // AddrOfOp) can have ASYNCHRONOUS/VOLATILE attributes even if the ultimate
    // entity (the globalOp) does not have them.
    if (auto addressOfOp = mlir::dyn_cast<fir::AddrOfOp>(definingOp)) {
      if (testAttributeSets(addressOfOp->getAttrs(), attributeNames))
        return true;
      if (auto module = definingOp->getParentOfType<mlir::ModuleOp>())
        if (auto globalOp =
                module.lookupSymbol<fir::GlobalOp>(addressOfOp.getSymbol()))
          return testAttributeSets(globalOp->getAttrs(), attributeNames);
    }
  }
  // TODO: Construct associated entities attributes. Decide where the fir
  // attributes must be placed/looked for in this case.
  return std::nullopt;
}

bool fir::valueMayHaveFirAttributes(
    mlir::Value value, llvm::ArrayRef<llvm::StringRef> attributeNames) {
  std::optional<bool> mayHaveAttr =
      valueCheckFirAttributes(value, attributeNames, /*checkAny=*/true);
  return mayHaveAttr.value_or(true);
}

bool fir::valueHasFirAttribute(mlir::Value value,
                               llvm::StringRef attributeName) {
  std::optional<bool> mayHaveAttr =
      valueCheckFirAttributes(value, {attributeName}, /*checkAny=*/false);
  return mayHaveAttr.value_or(false);
}

bool fir::anyFuncArgsHaveAttr(mlir::func::FuncOp func, llvm::StringRef attr) {
  for (unsigned i = 0, end = func.getNumArguments(); i < end; ++i)
    if (func.getArgAttr(i, attr))
      return true;
  return false;
}

std::optional<std::int64_t> fir::getIntIfConstant(mlir::Value value) {
  if (auto *definingOp = value.getDefiningOp()) {
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(definingOp))
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
        return intAttr.getInt();
    if (auto llConstOp = mlir::dyn_cast<mlir::LLVM::ConstantOp>(definingOp))
      if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(llConstOp.getValue()))
        return attr.getValue().getSExtValue();
  }
  return {};
}

bool fir::isDummyArgument(mlir::Value v) {
  auto blockArg{mlir::dyn_cast<mlir::BlockArgument>(v)};
  if (!blockArg) {
    auto defOp = v.getDefiningOp();
    if (defOp) {
      if (auto declareOp = mlir::dyn_cast<fir::DeclareOp>(defOp))
        if (declareOp.getDummyScope())
          return true;
    }
    return false;
  }

  auto *owner{blockArg.getOwner()};
  return owner->isEntryBlock() &&
         mlir::isa<mlir::FunctionOpInterface>(owner->getParentOp());
}

mlir::Type fir::applyPathToType(mlir::Type eleTy, mlir::ValueRange path) {
  for (auto i = path.begin(), end = path.end(); eleTy && i < end;) {
    eleTy = llvm::TypeSwitch<mlir::Type, mlir::Type>(eleTy)
                .Case<fir::RecordType>([&](fir::RecordType ty) {
                  if (auto *op = (*i++).getDefiningOp()) {
                    if (auto off = mlir::dyn_cast<fir::FieldIndexOp>(op))
                      return ty.getType(off.getFieldName());
                    if (auto off = mlir::dyn_cast<mlir::arith::ConstantOp>(op))
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
                    if (auto off = mlir::dyn_cast<mlir::arith::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
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

bool fir::reboxPreservesContinuity(fir::ReboxOp rebox, bool checkWhole) {
  // If slicing is not involved, then the rebox does not affect
  // the continuity of the array.
  auto sliceArg = rebox.getSlice();
  if (!sliceArg)
    return true;

  if (auto sliceOp =
          mlir::dyn_cast_or_null<fir::SliceOp>(sliceArg.getDefiningOp())) {
    if (sliceOp.getFields().empty() && sliceOp.getSubstr().empty()) {
      // TODO: generalize code for the triples analysis with
      // hlfir::designatePreservesContinuity, especially when
      // recognition of the whole dimension slices is added.
      auto triples = sliceOp.getTriples();
      assert((triples.size() % 3) == 0 && "invalid triples size");

      // A slice with step=1 in the innermost dimension preserves
      // the continuity of the array in the innermost dimension.
      // If checkWhole is false, then check only the innermost slice triples.
      std::size_t checkUpTo = checkWhole ? triples.size() : 3;
      checkUpTo = std::min(checkUpTo, triples.size());
      for (std::size_t i = 0; i < checkUpTo; i += 3) {
        if (triples[i] != triples[i + 1]) {
          // This is a section of the dimension. Only allow it
          // to be the first triple.
          if (i != 0)
            return false;
          auto constantStep = fir::getIntIfConstant(triples[i + 2]);
          if (!constantStep || *constantStep != 1)
            return false;
        }
      }
      return true;
    }
  }
  return false;
}

std::optional<int64_t> fir::getAllocaByteSize(fir::AllocaOp alloca,
                                              const mlir::DataLayout &dl,
                                              const fir::KindMapping &kindMap) {
  mlir::Type type = alloca.getInType();
  // TODO: should use the constant operands when all info is not available in
  // the type.
  if (!alloca.isDynamic())
    if (auto sizeAndAlignment =
            getTypeSizeAndAlignment(alloca.getLoc(), type, dl, kindMap))
      return sizeAndAlignment->first;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::DeclareOp::verify() {
  auto fortranVar =
      mlir::cast<fir::FortranVariableOpInterface>(this->getOperation());
  return fortranVar.verifyDeclareLikeOpImpl(getMemref());
}

//===----------------------------------------------------------------------===//
// PackArrayOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::PackArrayOp::verify() {
  mlir::Type arrayType = getArray().getType();
  if (!validTypeParams(arrayType, getTypeparams(), /*allowParamsForBox=*/true))
    return emitOpError("invalid type parameters");

  if (getInnermost() && fir::getBoxRank(arrayType) == 1)
    return emitOpError(
        "'innermost' is invalid for 1D arrays, use 'whole' instead");
  return mlir::success();
}

void fir::PackArrayOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  if (getStack())
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        mlir::SideEffects::AutomaticAllocationScopeResource::get());
  else
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(),
                         mlir::SideEffects::DefaultResource::get());

  if (!getNoCopy())
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
}

static mlir::ParseResult
parsePackArrayConstraints(mlir::OpAsmParser &parser, mlir::IntegerAttr &maxSize,
                          mlir::IntegerAttr &maxElementSize,
                          mlir::IntegerAttr &minStride) {
  mlir::OperationName opName = mlir::OperationName(
      fir::PackArrayOp::getOperationName(), parser.getContext());
  struct {
    llvm::StringRef name;
    mlir::IntegerAttr &ref;
  } attributes[] = {
      {fir::PackArrayOp::getMaxSizeAttrName(opName), maxSize},
      {fir::PackArrayOp::getMaxElementSizeAttrName(opName), maxElementSize},
      {fir::PackArrayOp::getMinStrideAttrName(opName), minStride}};

  mlir::NamedAttrList parsedAttrs;
  if (succeeded(parser.parseOptionalAttrDict(parsedAttrs))) {
    for (auto parsedAttr : parsedAttrs) {
      for (auto opAttr : attributes) {
        if (parsedAttr.getName() == opAttr.name)
          opAttr.ref = mlir::cast<mlir::IntegerAttr>(parsedAttr.getValue());
      }
    }
    return mlir::success();
  }
  return mlir::failure();
}

static void printPackArrayConstraints(mlir::OpAsmPrinter &p,
                                      fir::PackArrayOp &op,
                                      const mlir::IntegerAttr &maxSize,
                                      const mlir::IntegerAttr &maxElementSize,
                                      const mlir::IntegerAttr &minStride) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  if (maxSize)
    attributes.emplace_back(op.getMaxSizeAttrName(), maxSize);
  if (maxElementSize)
    attributes.emplace_back(op.getMaxElementSizeAttrName(), maxElementSize);
  if (minStride)
    attributes.emplace_back(op.getMinStrideAttrName(), minStride);

  p.printOptionalAttrDict(attributes);
}

//===----------------------------------------------------------------------===//
// UnpackArrayOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::UnpackArrayOp::verify() {
  if (auto packOp = getTemp().getDefiningOp<fir::PackArrayOp>())
    if (getStack() != packOp.getStack())
      return emitOpError() << "the pack operation uses different memory for "
                              "the temporary (stack vs heap): "
                           << *packOp.getOperation() << "\n";
  return mlir::success();
}

void fir::UnpackArrayOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  if (getStack())
    effects.emplace_back(
        mlir::MemoryEffects::Free::get(),
        mlir::SideEffects::AutomaticAllocationScopeResource::get());
  else
    effects.emplace_back(mlir::MemoryEffects::Free::get(),
                         mlir::SideEffects::DefaultResource::get());

  if (!getNoCopy())
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IsContiguousBoxOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyIsContiguousBoxOp
    : public mlir::OpRewritePattern<fir::IsContiguousBoxOp> {
  using mlir::OpRewritePattern<fir::IsContiguousBoxOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(fir::IsContiguousBoxOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace

mlir::LogicalResult SimplifyIsContiguousBoxOp::matchAndRewrite(
    fir::IsContiguousBoxOp op, mlir::PatternRewriter &rewriter) const {
  auto boxType = mlir::cast<fir::BaseBoxType>(op.getBox().getType());
  // Nothing to do for assumed-rank arrays and !fir.box<none>.
  if (boxType.isAssumedRank() || fir::isBoxNone(boxType))
    return mlir::failure();

  if (fir::getBoxRank(boxType) == 0) {
    // Scalars are always contiguous.
    mlir::Type i1Type = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, i1Type, rewriter.getIntegerAttr(i1Type, 1));
    return mlir::success();
  }

  // TODO: support more patterns, e.g. a result of fir.embox without
  // the slice is contiguous. We can add fir::isSimplyContiguous(box)
  // that walks def-use to figure it out.
  return mlir::failure();
}

void fir::IsContiguousBoxOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<SimplifyIsContiguousBoxOp>(context);
}

//===----------------------------------------------------------------------===//
// BoxTotalElementsOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyBoxTotalElementsOp
    : public mlir::OpRewritePattern<fir::BoxTotalElementsOp> {
  using mlir::OpRewritePattern<fir::BoxTotalElementsOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(fir::BoxTotalElementsOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace

mlir::LogicalResult SimplifyBoxTotalElementsOp::matchAndRewrite(
    fir::BoxTotalElementsOp op, mlir::PatternRewriter &rewriter) const {
  auto boxType = mlir::cast<fir::BaseBoxType>(op.getBox().getType());
  // Nothing to do for assumed-rank arrays and !fir.box<none>.
  if (boxType.isAssumedRank() || fir::isBoxNone(boxType))
    return mlir::failure();

  if (fir::getBoxRank(boxType) == 0) {
    // Scalar: 1 element.
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, op.getType(), rewriter.getIntegerAttr(op.getType(), 1));
    return mlir::success();
  }

  // TODO: support more cases, e.g. !fir.box<!fir.array<10xi32>>.
  return mlir::failure();
}

void fir::BoxTotalElementsOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<SimplifyBoxTotalElementsOp>(context);
}

//===----------------------------------------------------------------------===//
// LocalitySpecifierOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::LocalitySpecifierOp::verifyRegions() {
  mlir::Type argType = getArgType();
  auto verifyTerminator = [&](mlir::Operation *terminator,
                              bool yieldsValue) -> llvm::LogicalResult {
    if (!terminator->getBlock()->getSuccessors().empty())
      return llvm::success();

    if (!llvm::isa<fir::YieldOp>(terminator))
      return mlir::emitError(terminator->getLoc())
             << "expected exit block terminator to be an `fir.yield` op.";

    YieldOp yieldOp = llvm::cast<YieldOp>(terminator);
    mlir::TypeRange yieldedTypes = yieldOp.getResults().getTypes();

    if (!yieldsValue) {
      if (yieldedTypes.empty())
        return llvm::success();

      return mlir::emitError(terminator->getLoc())
             << "Did not expect any values to be yielded.";
    }

    if (yieldedTypes.size() == 1 && yieldedTypes.front() == argType)
      return llvm::success();

    auto error = mlir::emitError(yieldOp.getLoc())
                 << "Invalid yielded value. Expected type: " << argType
                 << ", got: ";

    if (yieldedTypes.empty())
      error << "None";
    else
      error << yieldedTypes;

    return error;
  };

  auto verifyRegion = [&](mlir::Region &region, unsigned expectedNumArgs,
                          llvm::StringRef regionName,
                          bool yieldsValue) -> llvm::LogicalResult {
    assert(!region.empty());

    if (region.getNumArguments() != expectedNumArgs)
      return mlir::emitError(region.getLoc())
             << "`" << regionName << "`: "
             << "expected " << expectedNumArgs
             << " region arguments, got: " << region.getNumArguments();

    for (mlir::Block &block : region) {
      // MLIR will verify the absence of the terminator for us.
      if (!block.mightHaveTerminator())
        continue;

      if (failed(verifyTerminator(block.getTerminator(), yieldsValue)))
        return llvm::failure();
    }

    return llvm::success();
  };

  // Ensure all of the region arguments have the same type
  for (mlir::Region *region : getRegions())
    for (mlir::Type ty : region->getArgumentTypes())
      if (ty != argType)
        return emitError() << "Region argument type mismatch: got " << ty
                           << " expected " << argType << ".";

  mlir::Region &initRegion = getInitRegion();
  if (!initRegion.empty() &&
      failed(verifyRegion(getInitRegion(), /*expectedNumArgs=*/2, "init",
                          /*yieldsValue=*/true)))
    return llvm::failure();

  LocalitySpecifierType dsType = getLocalitySpecifierType();

  if (dsType == LocalitySpecifierType::Local && !getCopyRegion().empty())
    return emitError("`local` specifiers do not require a `copy` region.");

  if (dsType == LocalitySpecifierType::LocalInit && getCopyRegion().empty())
    return emitError(
        "`local_init` specifiers require at least a `copy` region.");

  if (dsType == LocalitySpecifierType::LocalInit &&
      failed(verifyRegion(getCopyRegion(), /*expectedNumArgs=*/2, "copy",
                          /*yieldsValue=*/true)))
    return llvm::failure();

  if (!getDeallocRegion().empty() &&
      failed(verifyRegion(getDeallocRegion(), /*expectedNumArgs=*/1, "dealloc",
                          /*yieldsValue=*/false)))
    return llvm::failure();

  return llvm::success();
}

//===----------------------------------------------------------------------===//
// DoConcurrentOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::DoConcurrentOp::verify() {
  mlir::Block *body = getBody();

  if (body->empty())
    return emitOpError("body cannot be empty");

  if (!body->mightHaveTerminator() ||
      !mlir::isa<fir::DoConcurrentLoopOp>(body->getTerminator()))
    return emitOpError("must be terminated by 'fir.do_concurrent.loop'");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DoConcurrentLoopOp
//===----------------------------------------------------------------------===//

mlir::ParseResult fir::DoConcurrentLoopOp::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  llvm::SmallVector<mlir::OpAsmParser::Argument, 4> regionArgs;

  if (parser.parseArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes(regionArgs.size(),
                                         builder.getIndexType());

  // Parse loop bounds.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, regionArgs.size(),
                              mlir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return mlir::failure();

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, regionArgs.size(),
                              mlir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return mlir::failure();

  // Parse step values.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, regionArgs.size(),
                              mlir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return mlir::failure();

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> reduceOperands;
  llvm::SmallVector<mlir::Type> reduceArgTypes;
  if (succeeded(parser.parseOptionalKeyword("reduce"))) {
    // Parse reduction attributes and variables.
    llvm::SmallVector<fir::ReduceAttr> attributes;
    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::Paren, [&]() {
              if (parser.parseAttribute(attributes.emplace_back()) ||
                  parser.parseArrow() ||
                  parser.parseOperand(reduceOperands.emplace_back()) ||
                  parser.parseColonType(reduceArgTypes.emplace_back()))
                return mlir::failure();
              return mlir::success();
            })))
      return mlir::failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(reduceOperands, reduceArgTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return mlir::failure();
    llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                                 attributes.end());
    result.addAttribute(getReduceAttrsAttrName(result.name),
                        builder.getArrayAttr(arrayAttr));
  }

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> localOperands;
  if (succeeded(parser.parseOptionalKeyword("local"))) {
    std::size_t oldArgTypesSize = argTypes.size();
    if (failed(parser.parseLParen()))
      return mlir::failure();

    llvm::SmallVector<mlir::SymbolRefAttr> localSymbolVec;
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (failed(parser.parseAttribute(localSymbolVec.emplace_back())))
            return mlir::failure();

          if (parser.parseOperand(localOperands.emplace_back()) ||
              parser.parseArrow() ||
              parser.parseArgument(regionArgs.emplace_back()))
            return mlir::failure();

          return mlir::success();
        })))
      return mlir::failure();

    if (failed(parser.parseColon()))
      return mlir::failure();

    if (failed(parser.parseCommaSeparatedList([&]() {
          if (failed(parser.parseType(argTypes.emplace_back())))
            return mlir::failure();

          return mlir::success();
        })))
      return mlir::failure();

    if (regionArgs.size() != argTypes.size())
      return parser.emitError(parser.getNameLoc(),
                              "mismatch in number of local arg and types");

    if (failed(parser.parseRParen()))
      return mlir::failure();

    for (auto operandType : llvm::zip_equal(
             localOperands, llvm::drop_begin(argTypes, oldArgTypesSize)))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return mlir::failure();

    llvm::SmallVector<mlir::Attribute> symbolAttrs(localSymbolVec.begin(),
                                                   localSymbolVec.end());
    result.addAttribute(getLocalSymsAttrName(result.name),
                        builder.getArrayAttr(symbolAttrs));
  }

  // Set `operandSegmentSizes` attribute.
  result.addAttribute(DoConcurrentLoopOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(lower.size()),
                           static_cast<int32_t>(upper.size()),
                           static_cast<int32_t>(steps.size()),
                           static_cast<int32_t>(reduceOperands.size()),
                           static_cast<int32_t>(localOperands.size())}));

  // Now parse the body.
  for (auto [arg, type] : llvm::zip_equal(regionArgs, argTypes))
    arg.type = type;

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  return mlir::success();
}

void fir::DoConcurrentLoopOp::print(mlir::OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments().slice(0, getNumInductionVars())
    << ") = (" << getLowerBound() << ") to (" << getUpperBound() << ") step ("
    << getStep() << ")";

  if (!getReduceOperands().empty()) {
    p << " reduce(";
    auto attrs = getReduceAttrsAttr();
    auto operands = getReduceOperands();
    llvm::interleaveComma(llvm::zip(attrs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " -> " << std::get<1>(it) << " : "
        << std::get<1>(it).getType();
    });
    p << ')';
  }

  if (!getLocalVars().empty()) {
    p << " local(";
    llvm::interleaveComma(llvm::zip_equal(getLocalSymsAttr(), getLocalVars(),
                                          getRegionLocalArgs()),
                          p, [&](auto it) {
                            p << std::get<0>(it) << " " << std::get<1>(it)
                              << " -> " << std::get<2>(it);
                          });
    p << " : ";
    llvm::interleaveComma(getLocalVars(), p,
                          [&](auto it) { p << it.getType(); });
    p << ")";
  }

  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{DoConcurrentLoopOp::getOperandSegmentSizeAttr(),
                       DoConcurrentLoopOp::getReduceAttrsAttrName(),
                       DoConcurrentLoopOp::getLocalSymsAttrName()});
}

llvm::SmallVector<mlir::Region *> fir::DoConcurrentLoopOp::getLoopRegions() {
  return {&getRegion()};
}

llvm::LogicalResult fir::DoConcurrentLoopOp::verify() {
  mlir::Operation::operand_range lbValues = getLowerBound();
  mlir::Operation::operand_range ubValues = getUpperBound();
  mlir::Operation::operand_range stepValues = getStep();
  mlir::Operation::operand_range localVars = getLocalVars();

  if (lbValues.empty())
    return emitOpError(
        "needs at least one tuple element for lowerBound, upperBound and step");

  if (lbValues.size() != ubValues.size() ||
      ubValues.size() != stepValues.size())
    return emitOpError("different number of tuple elements for lowerBound, "
                       "upperBound or step");

  // Check that the body defines the same number of block arguments as the
  // number of tuple elements in step.
  mlir::Block *body = getBody();
  unsigned numIndVarArgs = body->getNumArguments() - localVars.size();

  if (numIndVarArgs != stepValues.size())
    return emitOpError() << "expects the same number of induction variables: "
                         << body->getNumArguments()
                         << " as bound and step values: " << stepValues.size();
  for (auto arg : body->getArguments().slice(0, numIndVarArgs))
    if (!arg.getType().isIndex())
      return emitOpError(
          "expects arguments for the induction variable to be of index type");

  auto reduceAttrs = getReduceAttrsAttr();
  if (getNumReduceOperands() != (reduceAttrs ? reduceAttrs.size() : 0))
    return emitOpError(
        "mismatch in number of reduction variables and reduction attributes");

  return mlir::success();
}

std::optional<llvm::SmallVector<mlir::Value>>
fir::DoConcurrentLoopOp::getLoopInductionVars() {
  return llvm::SmallVector<mlir::Value>{
      getBody()->getArguments().slice(0, getLowerBound().size())};
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void fir::FIROpsDialect::registerOpExternalInterfaces() {
  // Attach default declare target interfaces to operations which can be marked
  // as declare target.
  fir::GlobalOp::attachInterface<
      mlir::omp::DeclareTargetDefaultModel<fir::GlobalOp>>(*getContext());
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
