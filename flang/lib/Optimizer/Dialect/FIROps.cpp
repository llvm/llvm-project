//===-- FIROps.cpp --------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "aiir/Dialect/CommonFolders.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/TypeRange.h"
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

static void propagateAttributes(aiir::Operation *fromOp,
                                aiir::Operation *toOp) {
  if (!fromOp || !toOp)
    return;

  for (aiir::NamedAttribute attr : fromOp->getAttrs()) {
    if (attr.getName().getValue().starts_with(
            aiir::acc::OpenACCDialect::getDialectNamespace()))
      toOp->setAttr(attr.getName(), attr.getValue());
  }
}

/// Return true if a sequence type is of some incomplete size or a record type
/// is malformed or contains an incomplete sequence type. An incomplete sequence
/// type is one with more unknown extents in the type than have been provided
/// via `dynamicExtents`. Sequence types with an unknown rank are incomplete by
/// definition.
static bool verifyInType(aiir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited,
                         unsigned dynamicExtents = 0) {
  if (auto st = aiir::dyn_cast<fir::SequenceType>(inType)) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (std::size_t i = 0, end = shape.size(); i < end; ++i) {
      if (shape[i] != fir::SequenceType::getUnknownExtent())
        continue;
      if (dynamicExtents-- == 0)
        return true;
    }
  } else if (auto rt = aiir::dyn_cast<fir::RecordType>(inType)) {
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

static bool verifyTypeParamCount(aiir::Type inType, unsigned numParams) {
  auto ty = fir::unwrapSequenceType(inType);
  if (numParams > 0) {
    if (auto recTy = aiir::dyn_cast<fir::RecordType>(ty))
      return numParams != recTy.getNumLenParams();
    if (auto chrTy = aiir::dyn_cast<fir::CharacterType>(ty))
      return !(numParams == 1 && chrTy.hasDynamicLen());
    return true;
  }
  if (auto chrTy = aiir::dyn_cast<fir::CharacterType>(ty))
    return !chrTy.hasConstantLen();
  return false;
}

/// Parser shared by Alloca and Allocmem
/// operation ::= %res = (`fir.alloca` | `fir.allocmem`) $in_type
///                      ( `(` $typeparams `)` )? ( `,` $shape )?
///                      attr-dict-without-keyword
template <typename FN>
static aiir::ParseResult parseAllocatableOp(FN wrapResultType,
                                            aiir::OpAsmParser &parser,
                                            aiir::OperationState &result) {
  aiir::Type intype;
  if (parser.parseType(intype))
    return aiir::failure();
  auto &builder = parser.getBuilder();
  result.addAttribute("in_type", aiir::TypeAttr::get(intype));
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> operands;
  llvm::SmallVector<aiir::Type> typeVec;
  bool hasOperands = false;
  std::int32_t typeparamsSize = 0;
  if (!parser.parseOptionalLParen()) {
    // parse the LEN params of the derived type. (<params> : <types>)
    if (parser.parseOperandList(operands, aiir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(typeVec) || parser.parseRParen())
      return aiir::failure();
    typeparamsSize = operands.size();
    hasOperands = true;
  }
  std::int32_t shapeSize = 0;
  if (!parser.parseOptionalComma()) {
    // parse size to scale by, vector of n dimensions of type index
    if (parser.parseOperandList(operands, aiir::OpAsmParser::Delimiter::None))
      return aiir::failure();
    shapeSize = operands.size() - typeparamsSize;
    auto idxTy = builder.getIndexType();
    for (std::int32_t i = typeparamsSize, end = operands.size(); i != end; ++i)
      typeVec.push_back(idxTy);
    hasOperands = true;
  }
  if (hasOperands &&
      parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                             result.operands))
    return aiir::failure();
  aiir::Type restype = wrapResultType(intype);
  if (!restype) {
    parser.emitError(parser.getNameLoc(), "invalid allocate type: ") << intype;
    return aiir::failure();
  }
  result.addAttribute("operandSegmentSizes", builder.getDenseI32ArrayAttr(
                                                 {typeparamsSize, shapeSize}));
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(restype, result.types))
    return aiir::failure();
  return aiir::success();
}

template <typename OP>
static void printAllocatableOp(aiir::OpAsmPrinter &p, OP &op) {
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

bool fir::mayBeAbsentBox(aiir::Value val) {
  assert(aiir::isa<fir::BaseBoxType>(val.getType()) && "expected box argument");
  while (val) {
    aiir::Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return true;

    if (auto varIface = aiir::dyn_cast<fir::FortranVariableOpInterface>(defOp))
      return varIface.isOptional();

    // Check for fir.embox and fir.rebox before checking for
    // FortranObjectViewOpInterface, which they support.
    // A box created by fir.embox/rebox cannot be absent.
    if (aiir::isa<fir::ReboxOp, fir::EmboxOp, fir::LoadOp>(defOp))
      return false;

    if (auto viewIface =
            aiir::dyn_cast<fir::FortranObjectViewOpInterface>(defOp)) {
      val = viewIface.getViewSource(aiir::cast<aiir::OpResult>(val));
      continue;
    }
    break;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Create a legal memory reference as return type
static aiir::Type wrapAllocaResultType(aiir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (aiir::isa<fir::ReferenceType>(intype))
    return {};
  return fir::ReferenceType::get(intype);
}

llvm::SmallVector<aiir::MemorySlot> fir::AllocaOp::getPromotableSlots() {
  // TODO: support promotion of dynamic allocas
  if (isDynamic())
    return {};

  return {aiir::MemorySlot{getResult(), getAllocatedType()}};
}

aiir::Value fir::AllocaOp::getDefaultValue(const aiir::MemorySlot &slot,
                                           aiir::OpBuilder &builder) {
  return fir::UndefOp::create(builder, getLoc(), slot.elemType);
}

void fir::AllocaOp::handleBlockArgument(const aiir::MemorySlot &slot,
                                        aiir::BlockArgument argument,
                                        aiir::OpBuilder &builder) {
  // When there is a fir.declare, fir.debug_value must be emitted at each value
  // change and at each beginning of a block where the reaching value is
  // propagated as a block argument.
  // TODO: in order to get proper inter-dialect mem2reg, the
  // PromotableOpInterface should be provided with a
  // requiresInsertedBlockArguments similar to requiresReplacedValues so that
  // fir::DeclareOp can be the one dictating that this needs to happen instead
  // of the allocation. There are other challenges to inter dialect mem2reg to
  // solve first, like having a common concept for going through converts and
  // no-ops like fir.declare (i.e., to replace the FIR specific
  // isSlotOrDeclaredSlot).
  for (aiir::Operation *user : getOperation()->getUsers())
    if (auto declareOp = aiir::dyn_cast<fir::DeclareOp>(user))
      fir::DeclareValueOp::create(
          builder, declareOp.getLoc(), argument, declareOp.getDummyScope(),
          declareOp.getUniqNameAttr(), declareOp.getFortranAttrsAttr(),
          declareOp.getDataAttrAttr(), declareOp.getDummyArgNoAttr());
}

std::optional<aiir::PromotableAllocationOpInterface>
fir::AllocaOp::handlePromotionComplete(const aiir::MemorySlot &slot,
                                       aiir::Value defaultValue,
                                       aiir::OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty()) {
    assert(aiir::isa<fir::UndefOp>(defaultValue.getDefiningOp()) &&
           "Expected undef op to be the default value");
    defaultValue.getDefiningOp()->erase();
  }
  this->erase();
  return std::nullopt;
}

aiir::Type fir::AllocaOp::getAllocatedType() {
  return aiir::cast<fir::ReferenceType>(getType()).getEleTy();
}

aiir::Type fir::AllocaOp::getRefTy(aiir::Type ty) {
  return fir::ReferenceType::get(ty);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          llvm::StringRef uniqName, aiir::ValueRange typeparams,
                          aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          llvm::StringRef uniqName, bool pinned,
                          aiir::ValueRange typeparams, aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr, {},
        pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          aiir::ValueRange typeparams, aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? aiir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? aiir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          llvm::StringRef uniqName, llvm::StringRef bindcName,
                          bool pinned, aiir::ValueRange typeparams,
                          aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr =
      uniqName.empty() ? aiir::StringAttr{} : builder.getStringAttr(uniqName);
  auto bindcAttr =
      bindcName.empty() ? aiir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType), inType, nameAttr,
        bindcAttr, pinned, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          aiir::ValueRange typeparams, aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {},
        /*pinned=*/false, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocaOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Type inType,
                          bool pinned, aiir::ValueRange typeparams,
                          aiir::ValueRange shape,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocaResultType(inType), inType, {}, {}, pinned,
        typeparams, shape);
  result.addAttributes(attributes);
}

aiir::ParseResult fir::AllocaOp::parse(aiir::OpAsmParser &parser,
                                       aiir::OperationState &result) {
  return parseAllocatableOp(wrapAllocaResultType, parser, result);
}

void fir::AllocaOp::print(aiir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

llvm::LogicalResult fir::AllocaOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  aiir::Type outType = getType();
  if (!aiir::isa<fir::ReferenceType>(outType))
    return emitOpError("must be a !fir.ref type");
  return aiir::success();
}

bool fir::AllocaOp::ownsNestedAlloca(aiir::Operation *op) {
  return op->hasTrait<aiir::OpTrait::IsIsolatedFromAbove>() ||
         op->hasTrait<aiir::OpTrait::AutomaticAllocationScope>() ||
         aiir::isa<aiir::LoopLikeOpInterface>(*op);
}

aiir::Region *fir::AllocaOp::getOwnerRegion() {
  aiir::Operation *currentOp = getOperation();
  while (aiir::Operation *parentOp = currentOp->getParentOp()) {
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
static aiir::Type wrapAllocMemResultType(aiir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (aiir::isa<fir::ReferenceType, fir::HeapType, fir::PointerType,
                aiir::FunctionType>(intype))
    return {};
  return fir::HeapType::get(intype);
}

aiir::Type fir::AllocMemOp::getAllocatedType() {
  return aiir::cast<fir::HeapType>(getType()).getEleTy();
}

aiir::Type fir::AllocMemOp::getRefTy(aiir::Type ty) {
  return fir::HeapType::get(ty);
}

void fir::AllocMemOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result, aiir::Type inType,
                            llvm::StringRef uniqName,
                            aiir::ValueRange typeparams, aiir::ValueRange shape,
                            llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result, aiir::Type inType,
                            llvm::StringRef uniqName, llvm::StringRef bindcName,
                            aiir::ValueRange typeparams, aiir::ValueRange shape,
                            llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  auto nameAttr = builder.getStringAttr(uniqName);
  auto bindcAttr = builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocMemResultType(inType), inType, nameAttr,
        bindcAttr, typeparams, shape);
  result.addAttributes(attributes);
}

void fir::AllocMemOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result, aiir::Type inType,
                            aiir::ValueRange typeparams, aiir::ValueRange shape,
                            llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  build(builder, result, wrapAllocMemResultType(inType), inType, {}, {},
        typeparams, shape);
  result.addAttributes(attributes);
}

aiir::ParseResult fir::AllocMemOp::parse(aiir::OpAsmParser &parser,
                                         aiir::OperationState &result) {
  return parseAllocatableOp(wrapAllocMemResultType, parser, result);
}

void fir::AllocMemOp::print(aiir::OpAsmPrinter &p) {
  printAllocatableOp(p, *this);
}

llvm::LogicalResult fir::AllocMemOp::verify() {
  llvm::SmallVector<llvm::StringRef> visited;
  if (verifyInType(getInType(), visited, numShapeOperands()))
    return emitOpError("invalid type for allocation");
  if (verifyTypeParamCount(getInType(), numLenParams()))
    return emitOpError("LEN params do not correspond to type");
  aiir::Type outType = getType();
  if (!aiir::dyn_cast<fir::HeapType>(outType))
    return emitOpError("must be a !fir.heap type");
  if (fir::isa_unknown_size_box(fir::dyn_cast_ptrEleTy(outType)))
    return emitOpError("cannot allocate !fir.box of unknown rank or type");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ArrayCoorOp
//===----------------------------------------------------------------------===//

// CHARACTERs and derived types with LEN PARAMETERs are dependent types that
// require runtime values to fully define the type of an object.
static bool validTypeParams(aiir::Type dynTy, aiir::ValueRange typeParams,
                            bool allowParamsForBox = false) {
  dynTy = fir::unwrapAllRefAndSeqType(dynTy);
  if (aiir::isa<fir::BaseBoxType>(dynTy)) {
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
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(dynTy))
    return typeParams.size() == recTy.getNumLenParams();
  // Characters with non-constant LEN must have a type parameter value.
  if (auto charTy = aiir::dyn_cast<fir::CharacterType>(dynTy))
    if (charTy.hasDynamicLen())
      return typeParams.size() == 1;
  // Otherwise, any type parameters are invalid.
  return typeParams.size() == 0;
}

llvm::LogicalResult fir::ArrayCoorOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = aiir::dyn_cast<fir::SequenceType>(eleTy);
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = aiir::dyn_cast<fir::ShapeType>(shapeTy)) {
      shapeTyRank = s.getRank();
    } else if (auto ss = aiir::dyn_cast<fir::ShapeShiftType>(shapeTy)) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = aiir::cast<fir::ShiftType>(shapeTy);
      shapeTyRank = s.getRank();
      // TODO: it looks like PreCGRewrite and CodeGen can support
      // fir.shift with plain array reference, so we may consider
      // removing this check.
      if (!aiir::isa<fir::BaseBoxType>(getMemref().getType()))
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
    if (auto sl = aiir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_coor cannot take a slice with substring");
    if (auto sliceTy = aiir::dyn_cast<fir::SliceType>(sliceOp.getType()))
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }
  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");

  return aiir::success();
}

// Pull in fir.embox and fir.rebox into fir.array_coor when possible.
struct SimplifyArrayCoorOp : public aiir::OpRewritePattern<fir::ArrayCoorOp> {
  using aiir::OpRewritePattern<fir::ArrayCoorOp>::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(fir::ArrayCoorOp op,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::Value memref = op.getMemref();
    if (!aiir::isa<fir::BaseBoxType>(memref.getType()))
      return aiir::failure();

    aiir::Value boxedMemref, boxedShape, boxedSlice;
    if (auto emboxOp =
            aiir::dyn_cast_or_null<fir::EmboxOp>(memref.getDefiningOp())) {
      boxedMemref = emboxOp.getMemref();
      boxedShape = emboxOp.getShape();
      boxedSlice = emboxOp.getSlice();
      // If any of operands, that are not currently supported for migration
      // to ArrayCoorOp, is present, don't rewrite.
      if (!emboxOp.getTypeparams().empty() || emboxOp.getSourceBox() ||
          emboxOp.getAccessMap())
        return aiir::failure();
    } else if (auto reboxOp = aiir::dyn_cast_or_null<fir::ReboxOp>(
                   memref.getDefiningOp())) {
      // Don't pull in rebox when the array_coor is inside an ACC construct
      // and the rebox result is referenced by an ACC data clause.
      // The data legalization pipeline relies on the rebox result being the
      // copyin var; folding through it would leave the rebox source as an
      // unhandled live-in inside the compute region.
      if (op->getParentOfType<ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS>() &&
          llvm::any_of(memref.getUsers(), [](aiir::Operation *u) {
            return aiir::isa<ACC_DATA_ENTRY_OPS>(u);
          }))
        return aiir::failure();
      boxedMemref = reboxOp.getBox();
      boxedShape = reboxOp.getShape();
      // Avoid pulling in rebox that performs reshaping.
      // There is no way to represent box reshaping with array_coor.
      if (boxedShape && !aiir::isa<fir::ShiftType>(boxedShape.getType()))
        return aiir::failure();
      boxedSlice = reboxOp.getSlice();
    } else {
      return aiir::failure();
    }

    bool boxedShapeIsShift =
        boxedShape && aiir::isa<fir::ShiftType>(boxedShape.getType());
    bool boxedShapeIsShape =
        boxedShape && aiir::isa<fir::ShapeType>(boxedShape.getType());
    bool boxedShapeIsShapeShift =
        boxedShape && aiir::isa<fir::ShapeShiftType>(boxedShape.getType());

    // Slices changing the number of dimensions are not supported
    // for array_coor yet.
    unsigned origBoxRank;
    if (aiir::isa<fir::BaseBoxType>(boxedMemref.getType()))
      origBoxRank = fir::getBoxRank(boxedMemref.getType());
    else if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(
                 fir::unwrapRefType(boxedMemref.getType())))
      origBoxRank = arrTy.getDimension();
    else
      return aiir::failure();

    if (fir::getBoxRank(memref.getType()) != origBoxRank)
      return aiir::failure();

    // Slices with substring are not supported by array_coor.
    if (boxedSlice)
      if (auto sliceOp =
              aiir::dyn_cast_or_null<fir::SliceOp>(boxedSlice.getDefiningOp()))
        if (!sliceOp.getSubstr().empty())
          return aiir::failure();

    // If embox/rebox and array_coor have conflicting shapes or slices,
    // do nothing.
    if (op.getShape() && boxedShape && boxedShape != op.getShape())
      return aiir::failure();
    if (op.getSlice() && boxedSlice && boxedSlice != op.getSlice())
      return aiir::failure();

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
                return aiir::failure();
            } else {
              return aiir::failure();
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
              return aiir::failure();
            } else if (boxedShapeIsShape) {
              // %0 = fir.embox %arg(%shape) [%slice]
              // %1 = fir.array_coor %0 [%slice] %idx
              // This FIR may only be valid if the slice's start indices
              // and strides are all 1s.
              // We could pull in the embox as:
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
              return aiir::failure();
            } else if (boxedShapeIsShapeShift) {
              // %0 = fir.embox %arg(%shapeshift) [%slice]
              // %1 = fir.array_coor %0 [%slice] %idx
              // This FIR may only be valid if the shape specifies
              // that all lower bounds are 1s and the slice's start indices
              // and strides are all 1s.
              // We could pull in the embox as:
              // %shape = fir.shape <extents from the %shapeshift>
              // %1 = fir.array_coor %arg(%shape) [%slice] %idx
              return aiir::failure();
            } else {
              return aiir::failure();
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
                return aiir::failure();
            } else {
              return aiir::failure();
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
                return aiir::failure();
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
                return aiir::failure();
            } else {
              return aiir::failure();
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
            return aiir::failure();
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
            return aiir::failure();
          }
        }
      }
    }

    // TODO: temporarily avoid producing array_coor with the shape shift
    // and plain array reference (it seems to be a limitation of
    // ArrayCoorOp verifier).
    if (!aiir::isa<fir::BaseBoxType>(boxedMemref.getType())) {
      if (boxedShape) {
        if (aiir::isa<fir::ShiftType>(boxedShape.getType()))
          return aiir::failure();
      } else if (op.getShape() &&
                 aiir::isa<fir::ShiftType>(op.getShape().getType())) {
        return aiir::failure();
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
    return aiir::success();
  }

private:
  using IndicesVectorTy = std::vector<aiir::Value>;

  // If v is a shape_shift operation:
  //   fir.shape_shift %l1, %e1, %l2, %e2, ...
  // create:
  //   fir.shape %e1, %e2, ...
  static aiir::Value getShapeFromShapeShift(aiir::Value v,
                                            aiir::PatternRewriter &rewriter) {
    auto shapeShiftOp =
        aiir::dyn_cast_or_null<fir::ShapeShiftOp>(v.getDefiningOp());
    if (!shapeShiftOp)
      return nullptr;
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(shapeShiftOp);
    return fir::ShapeOp::create(rewriter, shapeShiftOp.getLoc(),
                                shapeShiftOp.getExtents());
  }

  static std::optional<IndicesVectorTy>
  getShiftedIndices(aiir::Value v, aiir::ValueRange indices,
                    aiir::PatternRewriter &rewriter) {
    auto insertAdjustments = [&](aiir::Operation *op, aiir::ValueRange lbs) {
      // Compute the shifted indices using the extended type.
      // Note that this can probably result in less efficient
      // AIIR and further LLVM IR due to the extra conversions.
      aiir::OpBuilder::InsertPoint savedIP = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(op);
      aiir::Location loc = op->getLoc();
      aiir::Type idxTy = rewriter.getIndexType();
      aiir::Value one = aiir::arith::ConstantOp::create(
          rewriter, loc, idxTy, rewriter.getIndexAttr(1));
      rewriter.restoreInsertionPoint(savedIP);
      auto nsw = aiir::arith::IntegerOverflowFlags::nsw;

      IndicesVectorTy shiftedIndices;
      for (auto [lb, idx] : llvm::zip(lbs, indices)) {
        aiir::Value extLb = fir::ConvertOp::create(rewriter, loc, idxTy, lb);
        aiir::Value extIdx = fir::ConvertOp::create(rewriter, loc, idxTy, idx);
        aiir::Value add =
            aiir::arith::AddIOp::create(rewriter, loc, extIdx, extLb, nsw);
        aiir::Value sub =
            aiir::arith::SubIOp::create(rewriter, loc, add, one, nsw);
        shiftedIndices.push_back(sub);
      }

      return shiftedIndices;
    };

    if (auto shiftOp =
            aiir::dyn_cast_or_null<fir::ShiftOp>(v.getDefiningOp())) {
      return insertAdjustments(shiftOp.getOperation(), shiftOp.getOrigins());
    } else if (auto shapeShiftOp = aiir::dyn_cast_or_null<fir::ShapeShiftOp>(
                   v.getDefiningOp())) {
      return insertAdjustments(shapeShiftOp.getOperation(),
                               shapeShiftOp.getOrigins());
    }

    return std::nullopt;
  }
};

void fir::ArrayCoorOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &patterns, aiir::AIIRContext *context) {
  // TODO: !fir.shape<1> operand may be removed from array_coor always.
  patterns.add<SimplifyArrayCoorOp>(context);
}

std::optional<std::int64_t> fir::ArrayCoorOp::getViewOffset(aiir::OpResult) {
  // TODO: we can try to compute the constant offset.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ArrayLoadOp
//===----------------------------------------------------------------------===//

static aiir::Type adjustedElementType(aiir::Type t) {
  if (auto ty = aiir::dyn_cast<fir::ReferenceType>(t)) {
    auto eleTy = ty.getEleTy();
    if (fir::isa_char(eleTy))
      return eleTy;
    if (fir::isa_derived(eleTy))
      return eleTy;
    if (aiir::isa<fir::SequenceType>(eleTy))
      return eleTy;
  }
  return t;
}

std::vector<aiir::Value> fir::ArrayLoadOp::getExtents() {
  if (auto sh = getShape())
    if (auto *op = sh.getDefiningOp()) {
      if (auto shOp = aiir::dyn_cast<fir::ShapeOp>(op)) {
        auto extents = shOp.getExtents();
        return {extents.begin(), extents.end()};
      }
      return aiir::cast<fir::ShapeShiftOp>(op).getExtents();
    }
  return {};
}

void fir::ArrayLoadOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(aiir::MemoryEffects::Read::get(), &getMemrefMutable(),
                       aiir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

llvm::LogicalResult fir::ArrayLoadOp::verify() {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  auto arrTy = aiir::dyn_cast<fir::SequenceType>(eleTy);
  if (!arrTy)
    return emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = getShape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0u;
    if (auto s = aiir::dyn_cast<fir::ShapeType>(shapeTy)) {
      shapeTyRank = s.getRank();
    } else if (auto ss = aiir::dyn_cast<fir::ShapeShiftType>(shapeTy)) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = aiir::cast<fir::ShiftType>(shapeTy);
      shapeTyRank = s.getRank();
      if (!aiir::isa<fir::BaseBoxType>(getMemref().getType()))
        return emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return emitOpError("rank of dimension mismatched");
  }

  if (auto sliceOp = getSlice()) {
    if (auto sl = aiir::dyn_cast_or_null<fir::SliceOp>(sliceOp.getDefiningOp()))
      if (!sl.getSubstr().empty())
        return emitOpError("array_load cannot take a slice with substring");
    if (auto sliceTy = aiir::dyn_cast<fir::SliceType>(sliceOp.getType()))
      if (sliceTy.getRank() != arrDim)
        return emitOpError("rank of dimension in slice mismatched");
  }

  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ArrayMergeStoreOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayMergeStoreOp::verify() {
  if (!aiir::isa<fir::ArrayLoadOp>(getOriginal().getDefiningOp()))
    return emitOpError("operand #0 must be result of a fir.array_load op");
  if (auto sl = getSlice()) {
    if (auto sliceOp =
            aiir::dyn_cast_or_null<fir::SliceOp>(sl.getDefiningOp())) {
      if (!sliceOp.getSubstr().empty())
        return emitOpError(
            "array_merge_store cannot take a slice with substring");
      if (!sliceOp.getFields().empty()) {
        // This is an intra-object merge, where the slice is projecting the
        // subfields that are to be overwritten by the merge operation.
        auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
        if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy)) {
          auto projTy =
              fir::applyPathToType(seqTy.getEleTy(), sliceOp.getFields());
          if (fir::unwrapSequenceType(getOriginal().getType()) != projTy)
            return emitOpError(
                "type of origin does not match sliced memref type");
          if (fir::unwrapSequenceType(getSequence().getType()) != projTy)
            return emitOpError(
                "type of sequence does not match sliced memref type");
          return aiir::success();
        }
        return emitOpError("referenced type is not an array");
      }
    }
    return aiir::success();
  }
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(getMemref().getType());
  if (getOriginal().getType() != eleTy)
    return emitOpError("type of origin does not match memref element type");
  if (getSequence().getType() != eleTy)
    return emitOpError("type of sequence does not match memref element type");
  if (!validTypeParams(getMemref().getType(), getTypeparams()))
    return emitOpError("invalid type parameters");
  return aiir::success();
}

void fir::ArrayMergeStoreOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(aiir::MemoryEffects::Write::get(), &getMemrefMutable(),
                       aiir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// ArrayFetchOp
//===----------------------------------------------------------------------===//

// Template function used for both array_fetch and array_update verification.
template <typename A>
aiir::Type validArraySubobject(A op) {
  auto ty = op.getSequence().getType();
  return fir::applyPathToType(ty, op.getIndices());
}

llvm::LogicalResult fir::ArrayFetchOp::verify() {
  auto arrTy = aiir::cast<fir::SequenceType>(getSequence().getType());
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      ::adjustedElementType(getElement().getType()) != arrTy.getEleTy())
    return emitOpError("return type does not match array");
  auto ty = validArraySubobject(*this);
  if (!ty || ty != ::adjustedElementType(getType()))
    return emitOpError("return type and/or indices do not type check");
  if (!aiir::isa<fir::ArrayLoadOp>(getSequence().getDefiningOp()))
    return emitOpError("argument #0 must be result of fir.array_load");
  if (!validTypeParams(arrTy, getTypeparams()))
    return emitOpError("invalid type parameters");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ArrayAccessOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayAccessOp::verify() {
  auto arrTy = aiir::cast<fir::SequenceType>(getSequence().getType());
  std::size_t indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices != dimension of array");
  if (indSize == arrTy.getDimension() &&
      getElement().getType() != fir::ReferenceType::get(arrTy.getEleTy()))
    return emitOpError("return type does not match array");
  aiir::Type ty = validArraySubobject(*this);
  if (!ty || fir::ReferenceType::get(ty) != getType())
    return emitOpError("return type and/or indices do not type check");
  if (!validTypeParams(arrTy, getTypeparams()))
    return emitOpError("invalid type parameters");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ArrayUpdateOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayUpdateOp::verify() {
  if (fir::isa_ref_type(getMerge().getType()))
    return emitOpError("does not support reference type for merge");
  auto arrTy = aiir::cast<fir::SequenceType>(getSequence().getType());
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
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ArrayModifyOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ArrayModifyOp::verify() {
  auto arrTy = aiir::cast<fir::SequenceType>(getSequence().getType());
  auto indSize = getIndices().size();
  if (indSize < arrTy.getDimension())
    return emitOpError("number of indices must match array dimension");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// BoxAddrOp
//===----------------------------------------------------------------------===//

void fir::BoxAddrOp::build(aiir::OpBuilder &builder,
                           aiir::OperationState &result, aiir::Value val) {
  aiir::Type type =
      llvm::TypeSwitch<aiir::Type, aiir::Type>(val.getType())
          .Case([&](fir::BaseBoxType ty) -> aiir::Type {
            aiir::Type eleTy = ty.getEleTy();
            if (fir::isa_ref_type(eleTy))
              return eleTy;
            return fir::ReferenceType::get(eleTy);
          })
          .Case([&](fir::BoxCharType ty) -> aiir::Type {
            return fir::ReferenceType::get(ty.getEleTy());
          })
          .Case([&](fir::BoxProcType ty) { return ty.getEleTy(); })
          .Default([&](const auto &) { return aiir::Type{}; });
  assert(type && "bad val type");
  build(builder, result, type, val);
}

aiir::OpFoldResult fir::BoxAddrOp::fold(FoldAdaptor adaptor) {
  if (auto *v = getVal().getDefiningOp()) {
    if (auto box = aiir::dyn_cast<fir::EmboxOp>(v)) {
      // Fold only if not sliced
      if (!box.getSlice() && box.getMemref().getType() == getType()) {
        propagateAttributes(getOperation(), box.getMemref().getDefiningOp());
        return box.getMemref();
      }
    }
    if (auto box = aiir::dyn_cast<fir::EmboxCharOp>(v))
      if (box.getMemref().getType() == getType())
        return box.getMemref();
  }
  return {};
}

std::optional<std::int64_t> fir::BoxAddrOp::getViewOffset(aiir::OpResult) {
  // fir.box_addr just returns the base address stored inside a box,
  // so the direct accesses through the base address and through the box
  // are not offsetted.
  return 0;
}

aiir::Speculation::Speculatability fir::BoxAddrOp::getSpeculatability() {
  // Do not speculate fir.box_addr with BoxProcType and BoxCharType
  // inputs.
  if (!aiir::isa<fir::BaseBoxType>(getVal().getType()))
    return aiir::Speculation::NotSpeculatable;
  return mayBeAbsentBox(getVal()) ? aiir::Speculation::NotSpeculatable
                                  : aiir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// BoxCharLenOp
//===----------------------------------------------------------------------===//

aiir::OpFoldResult fir::BoxCharLenOp::fold(FoldAdaptor adaptor) {
  if (auto v = getVal().getDefiningOp()) {
    if (auto box = aiir::dyn_cast<fir::EmboxCharOp>(v))
      return box.getLen();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxDimsOp
//===----------------------------------------------------------------------===//

/// Get the result types packed in a tuple tuple
aiir::Type fir::BoxDimsOp::getTupleType() {
  // note: triple, but 4 is nearest power of 2
  llvm::SmallVector<aiir::Type> triple{
      getResult(0).getType(), getResult(1).getType(), getResult(2).getType()};
  return aiir::TupleType::get(getContext(), triple);
}

aiir::Speculation::Speculatability fir::BoxDimsOp::getSpeculatability() {
  return mayBeAbsentBox(getVal()) ? aiir::Speculation::NotSpeculatable
                                  : aiir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// BoxRankOp
//===----------------------------------------------------------------------===//

void fir::BoxRankOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  aiir::OpOperand &inputBox = getBoxMutable();
  if (fir::isBoxAddress(inputBox.get().getType()))
    effects.emplace_back(aiir::MemoryEffects::Read::get(), &inputBox,
                         aiir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

aiir::FunctionType fir::CallOp::getFunctionType() {
  return aiir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

void fir::CallOp::print(aiir::OpAsmPrinter &p) {
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
  aiir::arith::FastMathFlagsAttr fmfAttr = getFastmathAttr();
  if (fmfAttr.getValue() != aiir::arith::FastMathFlags::none) {
    p << ' ' << aiir::arith::FastMathFlagsAttr::getMnemonic();
    p.printStrippedAttrOrType(fmfAttr);
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {fir::CallOp::getCalleeAttrNameStr(),
                           getFastmathAttrName(), getProcedureAttrsAttrName(),
                           getArgAttrsAttrName(), getResAttrsAttrName()});
  p << " : ";
  aiir::call_interface_impl::printFunctionSignature(
      p, getArgs().drop_front(isDirect ? 0 : 1).getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

aiir::ParseResult fir::CallOp::parse(aiir::OpAsmParser &parser,
                                     aiir::OperationState &result) {
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return aiir::failure();

  aiir::NamedAttrList attrs;
  aiir::SymbolRefAttr funcAttr;
  bool isDirect = operands.empty();
  if (isDirect)
    if (parser.parseAttribute(funcAttr, fir::CallOp::getCalleeAttrNameStr(),
                              attrs))
      return aiir::failure();

  if (parser.parseOperandList(operands, aiir::OpAsmParser::Delimiter::Paren))
    return aiir::failure();

  // Parse `proc_attrs<...>`, if present.
  fir::FortranProcedureFlagsEnumAttr procAttr;
  if (aiir::succeeded(parser.parseOptionalKeyword(
          fir::FortranProcedureFlagsEnumAttr::getMnemonic())))
    if (parser.parseCustomAttributeWithFallback(
            procAttr, aiir::Type{}, getProcedureAttrsAttrName(result.name),
            attrs))
      return aiir::failure();

  // Parse 'fastmath<...>', if present.
  aiir::arith::FastMathFlagsAttr fmfAttr;
  llvm::StringRef fmfAttrName = getFastmathAttrName(result.name);
  if (aiir::succeeded(parser.parseOptionalKeyword(fmfAttrName)))
    if (parser.parseCustomAttributeWithFallback(fmfAttr, aiir::Type{},
                                                fmfAttrName, attrs))
      return aiir::failure();

  if (parser.parseOptionalAttrDict(attrs) || parser.parseColon())
    return aiir::failure();
  llvm::SmallVector<aiir::Type> argTypes;
  llvm::SmallVector<aiir::Type> resTypes;
  llvm::SmallVector<aiir::DictionaryAttr> argAttrs;
  llvm::SmallVector<aiir::DictionaryAttr> resultAttrs;
  if (aiir::call_interface_impl::parseFunctionSignature(
          parser, argTypes, argAttrs, resTypes, resultAttrs))
    return parser.emitError(parser.getNameLoc(), "expected function type");
  aiir::FunctionType funcType =
      aiir::FunctionType::get(parser.getContext(), argTypes, resTypes);
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return aiir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<aiir::OpAsmParser::UnresolvedOperand>(operands)
            .drop_front();
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return aiir::failure();
  }
  result.attributes = attrs;
  aiir::call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  result.addTypes(funcType.getResults());
  return aiir::success();
}

void fir::CallOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                        aiir::func::FuncOp callee, aiir::ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute(getCalleeAttrNameStr(), aiir::SymbolRefAttr::get(callee));
  result.addTypes(callee.getFunctionType().getResults());
}

void fir::CallOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                        aiir::SymbolRefAttr callee,
                        llvm::ArrayRef<aiir::Type> results,
                        aiir::ValueRange operands) {
  result.addOperands(operands);
  if (callee)
    result.addAttribute(getCalleeAttrNameStr(), callee);
  result.addTypes(results);
}

void fir::CallOp::setCalleeFromCallable(aiir::CallInterfaceCallable callee) {
  if (auto symbolRef = llvm::dyn_cast<aiir::SymbolRefAttr>(callee)) {
    // Handling a direct call.
    bool wasIndirect = llvm::isa<aiir::Value>(getCallableForCallee());
    (*this)->setAttr(getCalleeAttrName(), symbolRef);
    // If it was indirect before, the operand list and associated attributes
    // needs to be fixed up.
    if (wasIndirect) {
      assert(getNumOperands() > 0 && "indirect call must have callee operand");
      (*this)->eraseOperand(0);
      // Fix arg_attrs to remove the first (callee) operand if needed.
      if (auto argAttrs = getArgAttrsAttr()) {
        // Since we already removed the first operand, check that number
        // of attributes is one more than number of operands.
        assert(argAttrs.size() == getNumOperands() + 1 &&
               "arg_attrs must be one-per-operand");
        llvm::SmallVector<aiir::Attribute> newAttrs(argAttrs.begin() + 1,
                                                    argAttrs.end());
        if (newAttrs.empty())
          (*this)->removeAttr(getArgAttrsAttrName());
        else
          (*this)->setAttr(getArgAttrsAttrName(),
                           aiir::ArrayAttr::get(getContext(), newAttrs));
      }
    }
    return;
  }
  // The provided callee makes this an indirect call now.
  bool wasIndirect = llvm::isa<aiir::Value>(getCallableForCallee());
  (*this)->removeAttr(getCalleeAttrNameStr());
  aiir::Value calleeVal = llvm::cast<aiir::Value>(callee);
  if (wasIndirect) {
    setOperand(0, calleeVal);
  } else {
    (*this)->insertOperands(0, calleeVal);
    // Make arg_attrs consistent in size with operands by adding an empty dict
    // for the callee.
    if (auto argAttrs = getArgAttrsAttr()) {
      assert(argAttrs.size() == getNumOperands() - 1 &&
             "arg_attrs must be one-per-operand");
      llvm::SmallVector<aiir::Attribute> newAttrs;
      newAttrs.reserve(1 + argAttrs.size());
      newAttrs.push_back(aiir::DictionaryAttr::get(getContext(), {}));
      newAttrs.append(argAttrs.begin(), argAttrs.end());
      (*this)->setAttr(getArgAttrsAttrName(),
                       aiir::ArrayAttr::get(getContext(), newAttrs));
    }
  }
}

//===----------------------------------------------------------------------===//
// CharConvertOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::CharConvertOp::verify() {
  auto unwrap = [&](aiir::Type t) {
    t = fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t));
    return aiir::dyn_cast<fir::CharacterType>(t);
  };
  auto inTy = unwrap(getFrom().getType());
  auto outTy = unwrap(getTo().getType());
  if (!(inTy && outTy))
    return emitOpError("not a reference to a character");
  if (inTy.getFKind() == outTy.getFKind())
    return emitOpError("buffers must have different KIND values");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

template <typename OPTY>
static void printCmpOp(aiir::OpAsmPrinter &p, OPTY op) {
  p << ' ';
  auto predSym = aiir::arith::symbolizeCmpFPredicate(
      op->template getAttrOfType<aiir::IntegerAttr>(
            OPTY::getPredicateAttrName())
          .getInt());
  assert(predSym.has_value() && "invalid symbol value for predicate");
  p << '"' << aiir::arith::stringifyCmpFPredicate(predSym.value()) << '"'
    << ", ";
  p.printOperand(op.getLhs());
  p << ", ";
  p.printOperand(op.getRhs());
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.getLhs().getType();
}

template <typename OPTY>
static aiir::ParseResult parseCmpOp(aiir::OpAsmParser &parser,
                                    aiir::OperationState &result) {
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> ops;
  aiir::NamedAttrList attrs;
  aiir::Attribute predicateNameAttr;
  aiir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return aiir::failure();

  if (!aiir::isa<aiir::StringAttr>(predicateNameAttr))
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      aiir::cast<aiir::StringAttr>(predicateNameAttr).getValue();
  auto predicate = fir::CmpcOp::getPredicateByName(predicateName);
  auto builder = parser.getBuilder();
  aiir::Type i1Type = builder.getI1Type();
  attrs.set(OPTY::getPredicateAttrName(),
            builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

static bool isBitcastCompatibleType(aiir::Type ty) {
  return aiir::isa<aiir::IntegerType, aiir::FloatType, fir::LogicalType>(ty) ||
         (aiir::isa<fir::CharacterType>(ty) &&
          aiir::cast<fir::CharacterType>(ty).getLen() ==
              fir::CharacterType::singleton());
}

static std::optional<unsigned> getBitcastBitSize(aiir::Type ty) {
  if (auto intTy = aiir::dyn_cast<aiir::IntegerType>(ty))
    return intTy.getWidth();
  if (auto floatTy = aiir::dyn_cast<aiir::FloatType>(ty))
    return floatTy.getWidth();
  // Bit size of fir.logical and fir.char depends on the kind map which is not
  // available in the verifier without an expensive lookup.
  return std::nullopt;
}

llvm::LogicalResult fir::BitcastOp::verify() {
  aiir::Type inType = getValue().getType();
  aiir::Type outType = getType();
  if (!isBitcastCompatibleType(inType))
    return emitOpError("input type is not bitcast compatible: ") << inType;
  if (!isBitcastCompatibleType(outType))
    return emitOpError("output type is not bitcast compatible: ") << outType;
  auto inBits = getBitcastBitSize(inType);
  auto outBits = getBitcastBitSize(outType);
  if (inBits && outBits && *inBits != *outBits)
    return emitOpError("bit size mismatch: input has ")
           << *inBits << " bits but output has " << *outBits << " bits";
  return aiir::success();
}

aiir::OpFoldResult fir::BitcastOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getType())
    return getValue();
  if (auto inner = getValue().getDefiningOp<fir::BitcastOp>()) {
    if (inner.getValue().getType() == getType())
      return inner.getValue();
    getValueMutable().assign(inner.getValue());
    return getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// CmpcOp
//===----------------------------------------------------------------------===//

void fir::buildCmpCOp(aiir::OpBuilder &builder, aiir::OperationState &result,
                      aiir::arith::CmpFPredicate predicate, aiir::Value lhs,
                      aiir::Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      fir::CmpcOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<std::int64_t>(predicate)));
}

aiir::arith::CmpFPredicate
fir::CmpcOp::getPredicateByName(llvm::StringRef name) {
  auto pred = aiir::arith::symbolizeCmpFPredicate(name);
  assert(pred.has_value() && "invalid predicate name");
  return pred.value();
}

void fir::CmpcOp::print(aiir::OpAsmPrinter &p) { printCmpOp(p, *this); }

aiir::ParseResult fir::CmpcOp::parse(aiir::OpAsmParser &parser,
                                     aiir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// VolatileCastOp
//===----------------------------------------------------------------------===//

static bool typesMatchExceptForVolatility(aiir::Type fromType,
                                          aiir::Type toType) {
  // If we can change only the volatility and get identical types, then we
  // match.
  if (fir::updateTypeWithVolatility(fromType, fir::isa_volatile_type(toType)) ==
      toType)
    return true;

  // Otherwise, recurse on the element types if the base classes are the same.
  const bool match =
      llvm::TypeSwitch<aiir::Type, bool>(fromType)
          .Case<fir::BoxType, fir::ReferenceType, fir::ClassType>(
              [&](auto type) {
                using TYPE = decltype(type);
                // If we are not the same base class, then we don't match.
                auto castedToType = aiir::dyn_cast<TYPE>(toType);
                if (!castedToType)
                  return false;
                // If we are the same base class, we match if the element types
                // match.
                return typesMatchExceptForVolatility(type.getEleTy(),
                                                     castedToType.getEleTy());
              })
          .Default([](aiir::Type) { return false; });

  return match;
}

llvm::LogicalResult fir::VolatileCastOp::verify() {
  aiir::Type fromType = getValue().getType();
  aiir::Type toType = getType();
  if (!typesMatchExceptForVolatility(fromType, toType))
    return emitOpError("types must be identical except for volatility ")
           << fromType << " / " << toType;
  return aiir::success();
}

aiir::OpFoldResult fir::VolatileCastOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getType())
    return getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void fir::ConvertOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.insert<ConvertConvertOptPattern, ConvertAscendingIndexOptPattern,
                 ConvertDescendingIndexOptPattern, RedundantConvertOptPattern,
                 CombineConvertOptPattern, CombineConvertTruncOptPattern,
                 ForwardConstantConvertPattern, ChainedPointerConvertsPattern>(
      context);
}

aiir::OpFoldResult fir::ConvertOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getType())
    return getValue();
  if (matchPattern(getValue(), aiir::m_Op<fir::ConvertOp>())) {
    auto inner = aiir::cast<fir::ConvertOp>(getValue().getDefiningOp());
    // (convert (convert 'a : logical -> i1) : i1 -> logical) ==> forward 'a
    if (auto toTy = aiir::dyn_cast<fir::LogicalType>(getType()))
      if (auto fromTy =
              aiir::dyn_cast<fir::LogicalType>(inner.getValue().getType()))
        if (aiir::isa<aiir::IntegerType>(inner.getType()) && (toTy == fromTy))
          return inner.getValue();
    // (convert (convert 'a : i1 -> logical) : logical -> i1) ==> forward 'a
    if (auto toTy = aiir::dyn_cast<aiir::IntegerType>(getType()))
      if (auto fromTy =
              aiir::dyn_cast<aiir::IntegerType>(inner.getValue().getType()))
        if (aiir::isa<fir::LogicalType>(inner.getType()) && (toTy == fromTy) &&
            (fromTy.getWidth() == 1))
          return inner.getValue();
  }
  return {};
}

bool fir::ConvertOp::isInteger(aiir::Type ty) {
  return aiir::isa<aiir::IntegerType, aiir::IndexType, fir::IntegerType>(ty);
}

bool fir::ConvertOp::isIntegerCompatible(aiir::Type ty) {
  return isInteger(ty) || aiir::isa<fir::LogicalType>(ty);
}

bool fir::ConvertOp::isFloatCompatible(aiir::Type ty) {
  return aiir::isa<aiir::FloatType>(ty);
}

bool fir::ConvertOp::isPointerCompatible(aiir::Type ty) {
  return aiir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
                   fir::LLVMPointerType, aiir::MemRefType, aiir::FunctionType,
                   fir::TypeDescType, aiir::LLVM::LLVMPointerType>(ty);
}

static std::optional<aiir::Type> getVectorElementType(aiir::Type ty) {
  aiir::Type elemTy;
  if (aiir::isa<fir::VectorType>(ty))
    elemTy = aiir::dyn_cast<fir::VectorType>(ty).getElementType();
  else if (aiir::isa<aiir::VectorType>(ty))
    elemTy = aiir::dyn_cast<aiir::VectorType>(ty).getElementType();
  else
    return std::nullopt;

  // e.g. fir.vector<4:ui32> => aiir.vector<4xi32>
  // e.g. aiir.vector<4xui32> => aiir.vector<4xi32>
  if (elemTy.isUnsignedInteger()) {
    elemTy = aiir::IntegerType::get(
        ty.getContext(), aiir::dyn_cast<aiir::IntegerType>(elemTy).getWidth());
  }
  return elemTy;
}

static std::optional<uint64_t> getVectorLen(aiir::Type ty) {
  if (aiir::isa<fir::VectorType>(ty))
    return aiir::dyn_cast<fir::VectorType>(ty).getLen();
  else if (aiir::isa<aiir::VectorType>(ty)) {
    // fir.vector only supports 1-D vector
    if (!(aiir::dyn_cast<aiir::VectorType>(ty).isScalable()))
      return aiir::dyn_cast<aiir::VectorType>(ty).getShape()[0];
  }

  return std::nullopt;
}

bool fir::ConvertOp::areVectorsCompatible(aiir::Type inTy, aiir::Type outTy) {
  if (!(aiir::isa<fir::VectorType>(inTy) &&
        aiir::isa<aiir::VectorType>(outTy)) &&
      !(aiir::isa<aiir::VectorType>(inTy) && aiir::isa<fir::VectorType>(outTy)))
    return false;

  // Only support integer, unsigned and real vector
  // Both vectors must have the same element type
  std::optional<aiir::Type> inElemTy = getVectorElementType(inTy);
  std::optional<aiir::Type> outElemTy = getVectorElementType(outTy);
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

static bool areRecordsCompatible(aiir::Type inTy, aiir::Type outTy) {
  // Both records must have the same field types.
  // Trust frontend semantics for in-depth checks, such as if both records
  // have the BIND(C) attribute.
  auto inRecTy = aiir::dyn_cast<fir::RecordType>(inTy);
  auto outRecTy = aiir::dyn_cast<fir::RecordType>(outTy);
  return inRecTy && outRecTy && inRecTy.getTypeList() == outRecTy.getTypeList();
}

bool fir::ConvertOp::canBeConverted(aiir::Type inType, aiir::Type outType) {
  if (inType == outType)
    return true;
  return (isPointerCompatible(inType) && isPointerCompatible(outType)) ||
         (isIntegerCompatible(inType) && isIntegerCompatible(outType)) ||
         (isInteger(inType) && isFloatCompatible(outType)) ||
         (isFloatCompatible(inType) && isInteger(outType)) ||
         (isFloatCompatible(inType) && isFloatCompatible(outType)) ||
         (isInteger(inType) && isPointerCompatible(outType)) ||
         (isPointerCompatible(inType) && isInteger(outType)) ||
         (aiir::isa<fir::BoxType>(inType) &&
          aiir::isa<fir::BoxType>(outType)) ||
         (aiir::isa<fir::BoxProcType>(inType) &&
          aiir::isa<fir::BoxProcType>(outType)) ||
         (fir::isa_complex(inType) && fir::isa_complex(outType)) ||
         (fir::isBoxedRecordType(inType) && fir::isPolymorphicType(outType)) ||
         (fir::isPolymorphicType(inType) && fir::isPolymorphicType(outType)) ||
         (fir::isPolymorphicType(inType) && aiir::isa<BoxType>(outType)) ||
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
static aiir::LogicalResult verifyVolatility(aiir::Type inType,
                                            aiir::Type outType) {
  const bool toLLVMPointer = aiir::isa<aiir::LLVM::LLVMPointerType>(outType);
  const bool toInteger = fir::isa_integer(outType);

  // When converting references to classes or allocatables into boxes for
  // runtime arguments, we cast away all the volatility information and pass a
  // box<none>. This is allowed.
  const bool isBoxNoneLike = [&]() {
    if (fir::isBoxNone(outType))
      return true;
    if (auto referenceType = aiir::dyn_cast<fir::ReferenceType>(outType)) {
      if (fir::isBoxNone(referenceType.getElementType())) {
        return true;
      }
    }
    return false;
  }();

  const bool isPtrToIntLike = toLLVMPointer || toInteger || isBoxNoneLike;
  if (isPtrToIntLike) {
    return aiir::success();
  }

  // In all other cases, we need to check for an exact volatility match.
  return aiir::success(fir::isa_volatile_type(inType) ==
                       fir::isa_volatile_type(outType));
}

llvm::LogicalResult fir::ConvertOp::verify() {
  aiir::Type inType = getValue().getType();
  aiir::Type outType = getType();
  if (fir::useStrictVolatileVerification()) {
    if (failed(verifyVolatility(inType, outType))) {
      return emitOpError("this conversion does not preserve volatility: ")
             << inType << " / " << outType;
    }
  }
  if (canBeConverted(inType, outType))
    return aiir::success();
  return emitOpError("invalid type conversion")
         << getValue().getType() << " / " << getType();
}

aiir::Speculation::Speculatability fir::ConvertOp::getSpeculatability() {
  // fir.convert is speculatable, in general. The only concern may be
  // converting from or/and to floating point types, which may trigger
  // some FP exceptions. Disallow speculating such converts for the time being.
  // Also disallow speculation for converts to/from non-FIR types, except
  // for some builtin types.
  auto canSpeculateType = [](aiir::Type ty) {
    if (fir::isa_fir_type(ty) || fir::isa_integer(ty))
      return true;
    return false;
  };
  return (canSpeculateType(getValue().getType()) && canSpeculateType(getType()))
             ? aiir::Speculation::Speculatable
             : aiir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// CoordinateOp
//===----------------------------------------------------------------------===//

void fir::CoordinateOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              aiir::Type resultType, aiir::Value ref,
                              aiir::ValueRange coor) {
  llvm::SmallVector<int32_t> fieldIndices;
  llvm::SmallVector<aiir::Value> dynamicIndices;
  bool anyField = false;
  for (aiir::Value index : coor) {
    if (auto field = index.getDefiningOp<fir::FieldIndexOp>()) {
      auto recTy = aiir::cast<fir::RecordType>(field.getOnType());
      fieldIndices.push_back(recTy.getFieldIndex(field.getFieldId()));
      anyField = true;
    } else {
      fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
      dynamicIndices.push_back(index);
    }
  }
  auto typeAttr = aiir::TypeAttr::get(ref.getType());
  if (anyField) {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr,
          builder.getDenseI32ArrayAttr(fieldIndices));
  } else {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr, nullptr);
  }
}

void fir::CoordinateOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              aiir::Type resultType, aiir::Value ref,
                              llvm::ArrayRef<fir::IntOrValue> coor) {
  llvm::SmallVector<int32_t> fieldIndices;
  llvm::SmallVector<aiir::Value> dynamicIndices;
  bool anyField = false;
  for (fir::IntOrValue index : coor) {
    llvm::TypeSwitch<fir::IntOrValue>(index)
        .Case([&](aiir::IntegerAttr intAttr) {
          fieldIndices.push_back(intAttr.getInt());
          anyField = true;
        })
        .Case([&](aiir::Value value) {
          dynamicIndices.push_back(value);
          fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
        });
  }
  auto typeAttr = aiir::TypeAttr::get(ref.getType());
  if (anyField) {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr,
          builder.getDenseI32ArrayAttr(fieldIndices));
  } else {
    build(builder, result, resultType, ref, dynamicIndices, typeAttr, nullptr);
  }
}

void fir::CoordinateOp::print(aiir::OpAsmPrinter &p) {
  p << ' ' << getRef();
  if (!getFieldIndicesAttr()) {
    p << ", " << getCoor();
  } else {
    aiir::Type eleTy = fir::getFortranElementType(getRef().getType());
    for (auto index : getIndices()) {
      p << ", ";
      llvm::TypeSwitch<fir::IntOrValue>(index)
          .Case([&](aiir::IntegerAttr intAttr) {
            if (auto recordType = llvm::dyn_cast<fir::RecordType>(eleTy)) {
              int fieldId = intAttr.getInt();
              if (fieldId < static_cast<int>(recordType.getNumFields())) {
                auto nameAndType = recordType.getTypeList()[fieldId];
                p << std::get<std::string>(nameAndType);
                eleTy = fir::getFortranElementType(
                    std::get<aiir::Type>(nameAndType));
                return;
              }
            }
            // Invalid index, still print it so that invalid IR can be
            // investigated.
            p << intAttr;
          })
          .Case([&](aiir::Value value) { p << value; });
    }
  }
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elideAttrs=*/{getBaseTypeAttrName(), getFieldIndicesAttrName()});
  p << " : ";
  p.printFunctionalType(getOperandTypes(), (*this)->getResultTypes());
}

aiir::ParseResult fir::CoordinateOp::parse(aiir::OpAsmParser &parser,
                                           aiir::OperationState &result) {
  aiir::OpAsmParser::UnresolvedOperand memref;
  if (parser.parseOperand(memref) || parser.parseComma())
    return aiir::failure();
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> coorOperands;
  llvm::SmallVector<std::pair<llvm::StringRef, int>> fieldNames;
  llvm::SmallVector<int32_t> fieldIndices;
  while (true) {
    llvm::StringRef fieldName;
    if (aiir::succeeded(parser.parseOptionalKeyword(&fieldName))) {
      fieldNames.push_back({fieldName, static_cast<int>(fieldIndices.size())});
      // Actual value will be computed later when base type has been parsed.
      fieldIndices.push_back(0);
    } else {
      aiir::OpAsmParser::UnresolvedOperand index;
      if (parser.parseOperand(index))
        return aiir::failure();
      fieldIndices.push_back(fir::CoordinateOp::kDynamicIndex);
      coorOperands.push_back(index);
    }
    if (aiir::failed(parser.parseOptionalComma()))
      break;
  }
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> allOperands;
  allOperands.push_back(memref);
  allOperands.append(coorOperands.begin(), coorOperands.end());
  aiir::FunctionType funcTy;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcTy) ||
      parser.resolveOperands(allOperands, funcTy.getInputs(), loc,
                             result.operands) ||
      parser.addTypesToList(funcTy.getResults(), result.types))
    return aiir::failure();
  result.addAttribute(getBaseTypeAttrName(result.name),
                      aiir::TypeAttr::get(funcTy.getInput(0)));
  if (!fieldNames.empty()) {
    aiir::Type eleTy = fir::getFortranElementType(funcTy.getInput(0));
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
          std::get<aiir::Type>(recTy.getTypeList()[fieldNum]));
    }
    result.addAttribute(getFieldIndicesAttrName(result.name),
                        parser.getBuilder().getDenseI32ArrayAttr(fieldIndices));
  }
  return aiir::success();
}

llvm::LogicalResult fir::CoordinateOp::verify() {
  const aiir::Type refTy = getRef().getType();
  if (fir::isa_ref_type(refTy)) {
    auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
    if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(eleTy)) {
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
    if (dimension == 0 && aiir::isa<fir::SequenceType>(eleTy)) {
      dimension = aiir::cast<fir::SequenceType>(eleTy).getDimension();
      if (dimension == 0)
        return emitOpError("cannot apply to array of unknown rank");
    }
    if (auto *defOp = co.getDefiningOp()) {
      if (auto index = aiir::dyn_cast<fir::LenParamIndexOp>(defOp)) {
        // Recovering a LEN type parameter only makes sense from a boxed
        // value. For a bare reference, the LEN type parameters must be
        // passed as additional arguments to `index`.
        if (aiir::isa<fir::BoxType>(refTy)) {
          if (coorOperand.index() != numCoors - 1)
            return emitOpError("len_param_index must be last argument");
          if (getNumOperands() != 2)
            return emitOpError("too many operands for len_param_index case");
        }
        if (eleTy != index.getOnType())
          return emitOpError(
              "len_param_index type not compatible with reference type");
        return aiir::success();
      } else if (auto index = aiir::dyn_cast<fir::FieldIndexOp>(defOp)) {
        if (eleTy != index.getOnType())
          return emitOpError(
              "field_index type not compatible with reference type");
        if (auto recTy = aiir::dyn_cast<fir::RecordType>(eleTy)) {
          eleTy = recTy.getType(index.getFieldName());
          continue;
        }
        return emitOpError("field_index not applied to !fir.type");
      }
    }
    if (dimension) {
      if (--dimension == 0)
        eleTy = aiir::cast<fir::SequenceType>(eleTy).getElementType();
    } else {
      if (auto t = aiir::dyn_cast<aiir::TupleType>(eleTy)) {
        // FIXME: Generally, we don't know which field of the tuple is being
        // referred to unless the operand is a constant. Just assume everything
        // is good in the tuple case for now.
        return aiir::success();
      } else if (auto t = aiir::dyn_cast<fir::RecordType>(eleTy)) {
        // FIXME: This is the same as the tuple case.
        return aiir::success();
      } else if (auto t = aiir::dyn_cast<aiir::ComplexType>(eleTy)) {
        eleTy = t.getElementType();
      } else if (auto t = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
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
  return aiir::success();
}

fir::CoordinateIndicesAdaptor fir::CoordinateOp::getIndices() {
  return CoordinateIndicesAdaptor(getFieldIndicesAttr(), getCoor());
}

std::optional<std::int64_t> fir::CoordinateOp::getViewOffset(aiir::OpResult) {
  // TODO: we can try to compute the constant offset.
  return std::nullopt;
}

aiir::Speculation::Speculatability fir::CoordinateOp::getSpeculatability() {
  const aiir::Type refTy = getRef().getType();
  if (fir::isa_ref_type(refTy))
    return aiir::Speculation::Speculatable;

  return mayBeAbsentBox(getRef()) ? aiir::Speculation::NotSpeculatable
                                  : aiir::Speculation::Speculatable;
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
  return aiir::success();
}

aiir::FunctionType fir::DispatchOp::getFunctionType() {
  return aiir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

//===----------------------------------------------------------------------===//
// TypeInfoOp
//===----------------------------------------------------------------------===//

void fir::TypeInfoOp::build(aiir::OpBuilder &builder,
                            aiir::OperationState &result, fir::RecordType type,
                            fir::RecordType parentType,
                            llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  result.addRegion();
  result.addRegion();
  result.addAttribute(aiir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(type.getName()));
  result.addAttribute(getTypeAttrName(result.name), aiir::TypeAttr::get(type));
  if (parentType)
    result.addAttribute(getParentTypeAttrName(result.name),
                        aiir::TypeAttr::get(parentType));
  result.addAttributes(attrs);
}

llvm::LogicalResult fir::TypeInfoOp::verify() {
  if (!getDispatchTable().empty())
    for (auto &op : getDispatchTable().front().without_terminator())
      if (!aiir::isa<fir::DTEntryOp>(op))
        return op.emitOpError("dispatch table must contain dt_entry");

  if (!aiir::isa<fir::RecordType>(getType()))
    return emitOpError("type must be a fir.type");

  if (getParentType() && !aiir::isa<fir::RecordType>(*getParentType()))
    return emitOpError("parent_type must be a fir.type");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// EmboxOp
//===----------------------------------------------------------------------===//

// Conversions from reference types to box types must preserve volatility.
static llvm::LogicalResult
verifyEmboxOpVolatilityInvariants(aiir::Type memrefType,
                                  aiir::Type resultType) {

  if (!fir::useStrictVolatileVerification())
    return aiir::success();

  aiir::Type boxElementType =
      llvm::TypeSwitch<aiir::Type, aiir::Type>(resultType)
          .Case<fir::BoxType, fir::ClassType>(
              [&](auto type) { return type.getEleTy(); })
          .Default([&](aiir::Type type) { return type; });

  // If the embox is simply wrapping a non-volatile type into a volatile box,
  // we're not losing any volatility information.
  if (boxElementType == memrefType) {
    return aiir::success();
  }

  // Otherwise, the volatility of the input and result must match.
  const bool volatilityMatches =
      fir::isa_volatile_type(memrefType) == fir::isa_volatile_type(resultType);

  return aiir::success(volatilityMatches);
}

llvm::LogicalResult fir::EmboxOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  bool isArray = false;
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy)) {
    eleTy = seqTy.getEleTy();
    isArray = true;
  }
  if (hasLenParams()) {
    auto lenPs = numLenParams();
    if (auto rt = aiir::dyn_cast<fir::RecordType>(eleTy)) {
      if (lenPs != rt.getNumLenParams())
        return emitOpError("number of LEN params does not correspond"
                           " to the !fir.type type");
    } else if (auto strTy = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
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
  if (getSourceBox() && !aiir::isa<fir::ClassType>(getResult().getType()))
    return emitOpError("source_box must be used with fir.class result type");
  if (failed(verifyEmboxOpVolatilityInvariants(getMemref().getType(),
                                               getResult().getType())))
    return emitOpError(
               "cannot convert between volatile and non-volatile types:")
           << " " << getMemref().getType() << " " << getResult().getType();
  return aiir::success();
}

/// Returns true if \p extent matches the extent of the \p box's
/// dimension \p dim.
static bool isBoxExtent(aiir::Value box, std::int64_t dim, aiir::Value extent) {
  if (auto op = extent.getDefiningOp<fir::BoxDimsOp>())
    if (op.getVal() == box && op.getExtent() == extent)
      if (auto dimOperand = fir::getIntIfConstant(op.getDim()))
        return *dimOperand == dim;
  return false;
}

/// Returns true if \p lb matches the lower bound of the \p box's
/// dimension \p dim. If \p mayHaveNonDefaultLowerBounds is false,
/// then \p lb may be an integer constant 1.
static bool isBoxLb(aiir::Value box, std::int64_t dim, aiir::Value lb,
                    bool mayHaveNonDefaultLowerBounds = true) {
  if (auto op = lb.getDefiningOp<fir::BoxDimsOp>()) {
    if (op.getVal() == box && op.getLowerBound() == lb)
      if (auto dimOperand = fir::getIntIfConstant(op.getDim()))
        return *dimOperand == dim;
  } else if (!mayHaveNonDefaultLowerBounds) {
    if (auto constantLb = fir::getIntIfConstant(lb))
      return *constantLb == 1;
  }
  return false;
}

/// Returns true if \p ub matches the upper bound of the \p box's
/// dimension \p dim. If \p mayHaveNonDefaultLowerBounds is false,
/// then the dimension's lower bound may be an integer constant 1.
/// Note that the upper bound is usually a result of computation
/// involving the lower bound and the extent, and the function
/// tries its best to recognize the computation pattern.
/// The conservative result 'false' does not necessarily mean
/// that \p ub is not an actual upper bound value.
static bool isBoxUb(aiir::Value box, std::int64_t dim, aiir::Value ub,
                    bool mayHaveNonDefaultLowerBounds = true) {
  if (auto sub1 = ub.getDefiningOp<aiir::arith::SubIOp>()) {
    auto one = fir::getIntIfConstant(sub1.getOperand(1));
    if (!one || *one != 1)
      return false;
    if (auto add = sub1.getOperand(0).getDefiningOp<aiir::arith::AddIOp>())
      if ((isBoxLb(box, dim, add.getOperand(0)) &&
           isBoxExtent(box, dim, add.getOperand(1))) ||
          (isBoxLb(box, dim, add.getOperand(1)) &&
           isBoxExtent(box, dim, add.getOperand(0))))
        return true;
  } else if (!mayHaveNonDefaultLowerBounds) {
    return isBoxExtent(box, dim, ub);
  }
  return false;
}

/// Checks if the given \p sliceOp specifies a contiguous
/// array slice. If \p checkWhole is true, then the check
/// is done for all dimensions, otherwise, only for the innermost
/// dimension.
/// The simplest way to prove that this is an contiguous slice
/// is to check whether the slice stride(s) is 1.
/// For more complex cases, extra information must be provided
/// by the caller:
///   * \p origBox - if not null, then the source array is represented
///     with this !fir.box value. The box is used to recognize
///     the full dimension slices, which are specified by the triplets
///     computed from the dimensions' lower bounds and extents.
///   * \p mayHaveNonDefaultLowerBounds may be set to false to indicate
///     that the source entity has default lower bounds, so the full
///     dimension slices computations may use 1 for the lower bound.
static bool isContiguousArraySlice(fir::SliceOp sliceOp, bool checkWhole = true,
                                   aiir::Value origBox = nullptr,
                                   bool mayHaveNonDefaultLowerBounds = true) {
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
        // to be the first triple, if the source of the slice
        // is a boxed array. If it is a raw pointer, then
        // the result will still be contiguous, as long as
        // the strides are all ones.
        // When origBox is not null, we must prove that the triple
        // covers the whole dimension and the stride is one,
        // before claiming contiguity for this dimension.
        if (i != 0 && origBox) {
          std::int64_t dim = i / 3;
          if (!isBoxLb(origBox, dim, triples[i],
                       mayHaveNonDefaultLowerBounds) ||
              !isBoxUb(origBox, dim, triples[i + 1],
                       mayHaveNonDefaultLowerBounds))
            return false;
        }
        auto constantStep = fir::getIntIfConstant(triples[i + 2]);
        if (!constantStep || *constantStep != 1)
          return false;
      }
    }
    return true;
  }
  return false;
}

bool fir::isContiguousEmbox(fir::EmboxOp embox, bool checkWhole) {
  auto sliceArg = embox.getSlice();
  if (!sliceArg)
    return true;

  if (auto sliceOp =
          aiir::dyn_cast_or_null<fir::SliceOp>(sliceArg.getDefiningOp()))
    return isContiguousArraySlice(sliceOp, checkWhole);

  return false;
}

std::optional<std::int64_t> fir::EmboxOp::getViewOffset(aiir::OpResult) {
  // The address offset is zero, unless there is a slice.
  // TODO: we can handle slices that leave the base address untouched.
  if (!getSlice())
    return 0;
  return std::nullopt;
}

aiir::Speculation::Speculatability fir::EmboxOp::getSpeculatability() {
  return (getSourceBox() && mayBeAbsentBox(getSourceBox()))
             ? aiir::Speculation::NotSpeculatable
             : aiir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// EmboxCharOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::EmboxCharOp::verify() {
  auto eleTy = fir::dyn_cast_ptrEleTy(getMemref().getType());
  if (!aiir::dyn_cast_or_null<fir::CharacterType>(eleTy))
    return aiir::failure();
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// EmboxProcOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::EmboxProcOp::verify() {
  // host bindings (optional) must be a reference to a tuple
  if (auto h = getHost()) {
    if (auto r = aiir::dyn_cast<fir::ReferenceType>(h.getType()))
      if (aiir::isa<aiir::TupleType>(r.getEleTy()))
        return aiir::success();
    return aiir::failure();
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// TypeDescOp
//===----------------------------------------------------------------------===//

void fir::TypeDescOp::build(aiir::OpBuilder &, aiir::OperationState &result,
                            aiir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

aiir::ParseResult fir::TypeDescOp::parse(aiir::OpAsmParser &parser,
                                         aiir::OperationState &result) {
  aiir::Type intype;
  if (parser.parseType(intype))
    return aiir::failure();
  result.addAttribute("in_type", aiir::TypeAttr::get(intype));
  aiir::Type restype = fir::TypeDescType::get(intype);
  if (parser.addTypeToList(restype, result.types))
    return aiir::failure();
  return aiir::success();
}

void fir::TypeDescOp::print(aiir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr("in_type");
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"in_type"});
}

llvm::LogicalResult fir::TypeDescOp::verify() {
  aiir::Type resultTy = getType();
  if (auto tdesc = aiir::dyn_cast<fir::TypeDescType>(resultTy)) {
    if (tdesc.getOfTy() != getInType())
      return emitOpError("wrapped type mismatched");
    return aiir::success();
  }
  return emitOpError("must be !fir.tdesc type");
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

aiir::Type fir::GlobalOp::resultType() {
  return wrapAllocaResultType(getType());
}

aiir::ParseResult fir::GlobalOp::parse(aiir::OpAsmParser &parser,
                                       aiir::OperationState &result) {
  // Parse the optional linkage
  llvm::StringRef linkage;
  auto &builder = parser.getBuilder();
  if (aiir::succeeded(parser.parseOptionalKeyword(&linkage))) {
    if (fir::GlobalOp::verifyValidLinkage(linkage))
      return aiir::failure();
    aiir::StringAttr linkAttr = builder.getStringAttr(linkage);
    result.addAttribute(fir::GlobalOp::getLinkNameAttrName(result.name),
                        linkAttr);
  }

  // Parse the name as a symbol reference attribute.
  aiir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr,
                            fir::GlobalOp::getSymrefAttrName(result.name),
                            result.attributes))
    return aiir::failure();
  result.addAttribute(aiir::SymbolTable::getSymbolAttrName(),
                      nameAttr.getRootReference());

  bool simpleInitializer = false;
  if (aiir::succeeded(parser.parseOptionalLParen())) {
    aiir::Attribute attr;
    if (parser.parseAttribute(attr, getInitValAttrName(result.name),
                              result.attributes) ||
        parser.parseRParen())
      return aiir::failure();
    simpleInitializer = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return aiir::failure();

  if (succeeded(
          parser.parseOptionalKeyword(getConstantAttrName(result.name)))) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  }

  if (succeeded(parser.parseOptionalKeyword(getTargetAttrName(result.name))))
    result.addAttribute(getTargetAttrName(result.name), builder.getUnitAttr());

  aiir::Type globalType;
  if (parser.parseColonType(globalType))
    return aiir::failure();

  result.addAttribute(fir::GlobalOp::getTypeAttrName(result.name),
                      aiir::TypeAttr::get(globalType));

  if (simpleInitializer) {
    result.addRegion();
  } else {
    // Parse the optional initializer body.
    auto parseResult =
        parser.parseOptionalRegion(*result.addRegion(), /*arguments=*/{});
    if (parseResult.has_value() && aiir::failed(*parseResult))
      return aiir::failure();
  }
  return aiir::success();
}

void fir::GlobalOp::print(aiir::OpAsmPrinter &p) {
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

void fir::GlobalOp::appendInitialValue(aiir::Operation *op) {
  getBlock().getOperations().push_back(op);
}

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, aiir::Type type,
                          aiir::Attribute initialVal, aiir::StringAttr linkage,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(getTypeAttrName(result.name), aiir::TypeAttr::get(type));
  result.addAttribute(aiir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getSymrefAttrName(result.name),
                      aiir::SymbolRefAttr::get(builder.getContext(), name));
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

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          aiir::Type type, aiir::Attribute initialVal,
                          aiir::StringAttr linkage,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        {}, linkage, attrs);
}

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, aiir::Type type,
                          aiir::StringAttr linkage,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, isTarget, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          aiir::Type type, aiir::StringAttr linkage,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        {}, linkage, attrs);
}

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          bool isConstant, bool isTarget, aiir::Type type,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  build(builder, result, name, isConstant, isTarget, type, aiir::StringAttr{},
        attrs);
}

void fir::GlobalOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, llvm::StringRef name,
                          aiir::Type type,
                          llvm::ArrayRef<aiir::NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, /*isTarget=*/false, type,
        attrs);
}

aiir::ParseResult fir::GlobalOp::verifyValidLinkage(llvm::StringRef linkage) {
  // Supporting only a subset of the LLVM linkage types for now
  static const char *validNames[] = {"common", "internal", "linkonce",
                                     "linkonce_odr", "weak"};
  return aiir::success(llvm::is_contained(validNames, linkage));
}

//===----------------------------------------------------------------------===//
// GlobalLenOp
//===----------------------------------------------------------------------===//

aiir::ParseResult fir::GlobalLenOp::parse(aiir::OpAsmParser &parser,
                                          aiir::OperationState &result) {
  llvm::StringRef fieldName;
  if (failed(parser.parseOptionalKeyword(&fieldName))) {
    aiir::StringAttr fieldAttr;
    if (parser.parseAttribute(fieldAttr,
                              fir::GlobalLenOp::getLenParamAttrName(),
                              result.attributes))
      return aiir::failure();
  } else {
    result.addAttribute(fir::GlobalLenOp::getLenParamAttrName(),
                        parser.getBuilder().getStringAttr(fieldName));
  }
  aiir::IntegerAttr constant;
  if (parser.parseComma() ||
      parser.parseAttribute(constant, fir::GlobalLenOp::getIntAttrName(),
                            result.attributes))
    return aiir::failure();
  return aiir::success();
}

void fir::GlobalLenOp::print(aiir::OpAsmPrinter &p) {
  p << ' ' << getOperation()->getAttr(fir::GlobalLenOp::getLenParamAttrName())
    << ", " << getOperation()->getAttr(fir::GlobalLenOp::getIntAttrName());
}

//===----------------------------------------------------------------------===//
// FieldIndexOp
//===----------------------------------------------------------------------===//

template <typename TY>
aiir::ParseResult parseFieldLikeOp(aiir::OpAsmParser &parser,
                                   aiir::OperationState &result) {
  llvm::StringRef fieldName;
  auto &builder = parser.getBuilder();
  aiir::Type recty;
  if (parser.parseOptionalKeyword(&fieldName) || parser.parseComma() ||
      parser.parseType(recty))
    return aiir::failure();
  result.addAttribute(fir::FieldIndexOp::getFieldAttrName(),
                      builder.getStringAttr(fieldName));
  if (!aiir::dyn_cast<fir::RecordType>(recty))
    return aiir::failure();
  result.addAttribute(fir::FieldIndexOp::getTypeAttrName(),
                      aiir::TypeAttr::get(recty));
  if (!parser.parseOptionalLParen()) {
    llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<aiir::Type> types;
    auto loc = parser.getNameLoc();
    if (parser.parseOperandList(operands, aiir::OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(types) || parser.parseRParen() ||
        parser.resolveOperands(operands, types, loc, result.operands))
      return aiir::failure();
  }
  aiir::Type fieldType = TY::get(builder.getContext());
  if (parser.addTypeToList(fieldType, result.types))
    return aiir::failure();
  return aiir::success();
}

aiir::ParseResult fir::FieldIndexOp::parse(aiir::OpAsmParser &parser,
                                           aiir::OperationState &result) {
  return parseFieldLikeOp<fir::FieldType>(parser, result);
}

template <typename OP>
void printFieldLikeOp(aiir::OpAsmPrinter &p, OP &op) {
  p << ' '
    << op.getOperation()
           ->template getAttrOfType<aiir::StringAttr>(
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

void fir::FieldIndexOp::print(aiir::OpAsmPrinter &p) {
  printFieldLikeOp(p, *this);
}

void fir::FieldIndexOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              llvm::StringRef fieldName, aiir::Type recTy,
                              aiir::ValueRange operands) {
  result.addAttribute(getFieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(getTypeAttrName(), aiir::TypeAttr::get(recTy));
  result.addOperands(operands);
}

llvm::SmallVector<aiir::Attribute> fir::FieldIndexOp::getAttributes() {
  llvm::SmallVector<aiir::Attribute> attrs;
  attrs.push_back(getFieldIdAttr());
  attrs.push_back(getOnTypeAttr());
  return attrs;
}

//===----------------------------------------------------------------------===//
// InsertOnRangeOp
//===----------------------------------------------------------------------===//

static aiir::ParseResult
parseCustomRangeSubscript(aiir::OpAsmParser &parser,
                          aiir::DenseIntElementsAttr &coord) {
  llvm::SmallVector<std::int64_t> lbounds;
  llvm::SmallVector<std::int64_t> ubounds;
  if (parser.parseKeyword("from") ||
      parser.parseCommaSeparatedList(
          aiir::AsmParser::Delimiter::Paren,
          [&] { return parser.parseInteger(lbounds.emplace_back(0)); }) ||
      parser.parseKeyword("to") ||
      parser.parseCommaSeparatedList(aiir::AsmParser::Delimiter::Paren, [&] {
        return parser.parseInteger(ubounds.emplace_back(0));
      }))
    return aiir::failure();
  llvm::SmallVector<std::int64_t> zippedBounds;
  for (auto zip : llvm::zip(lbounds, ubounds)) {
    zippedBounds.push_back(std::get<0>(zip));
    zippedBounds.push_back(std::get<1>(zip));
  }
  coord = aiir::Builder(parser.getContext()).getIndexTensorAttr(zippedBounds);
  return aiir::success();
}

static void printCustomRangeSubscript(aiir::OpAsmPrinter &printer,
                                      fir::InsertOnRangeOp op,
                                      aiir::DenseIntElementsAttr coord) {
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
  aiir::DenseIntElementsAttr coorAttr = getCoor();
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
  return aiir::success();
}

bool fir::InsertOnRangeOp::isFullRange() {
  auto extents = getType().getShape();
  aiir::DenseIntElementsAttr indexes = getCoor();
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

static bool checkIsIntegerConstant(aiir::Attribute attr, std::int64_t conVal) {
  if (auto iattr = aiir::dyn_cast<aiir::IntegerAttr>(attr))
    return iattr.getInt() == conVal;
  return false;
}

static bool isZero(aiir::Attribute a) { return checkIsIntegerConstant(a, 0); }
static bool isOne(aiir::Attribute a) { return checkIsIntegerConstant(a, 1); }

// Undo some complex patterns created in the front-end and turn them back into
// complex ops.
template <typename FltOp, typename CpxOp>
struct UndoComplexPattern : public aiir::RewritePattern {
  UndoComplexPattern(aiir::AIIRContext *ctx)
      : aiir::RewritePattern("fir.insert_value", 2, ctx) {}

  llvm::LogicalResult
  matchAndRewrite(aiir::Operation *op,
                  aiir::PatternRewriter &rewriter) const override {
    auto insval = aiir::dyn_cast_or_null<fir::InsertValueOp>(op);
    if (!insval || !aiir::isa<aiir::ComplexType>(insval.getType()))
      return aiir::failure();
    auto insval2 = aiir::dyn_cast_or_null<fir::InsertValueOp>(
        insval.getAdt().getDefiningOp());
    if (!insval2)
      return aiir::failure();
    auto binf = aiir::dyn_cast_or_null<FltOp>(insval.getVal().getDefiningOp());
    auto binf2 =
        aiir::dyn_cast_or_null<FltOp>(insval2.getVal().getDefiningOp());
    if (!binf || !binf2 || insval.getCoor().size() != 1 ||
        !isOne(insval.getCoor()[0]) || insval2.getCoor().size() != 1 ||
        !isZero(insval2.getCoor()[0]))
      return aiir::failure();
    auto eai = aiir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getLhs().getDefiningOp());
    auto ebi = aiir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf.getRhs().getDefiningOp());
    auto ear = aiir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getLhs().getDefiningOp());
    auto ebr = aiir::dyn_cast_or_null<fir::ExtractValueOp>(
        binf2.getRhs().getDefiningOp());
    if (!eai || !ebi || !ear || !ebr || ear.getAdt() != eai.getAdt() ||
        ebr.getAdt() != ebi.getAdt() || eai.getCoor().size() != 1 ||
        !isOne(eai.getCoor()[0]) || ebi.getCoor().size() != 1 ||
        !isOne(ebi.getCoor()[0]) || ear.getCoor().size() != 1 ||
        !isZero(ear.getCoor()[0]) || ebr.getCoor().size() != 1 ||
        !isZero(ebr.getCoor()[0]))
      return aiir::failure();
    rewriter.replaceOpWithNewOp<CpxOp>(op, ear.getAdt(), ebr.getAdt());
    return aiir::success();
  }
};

void fir::InsertValueOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &results, aiir::AIIRContext *context) {
  results.insert<UndoComplexPattern<aiir::arith::AddFOp, fir::AddcOp>,
                 UndoComplexPattern<aiir::arith::SubFOp, fir::SubcOp>>(context);
}

//===----------------------------------------------------------------------===//
// IterWhileOp
//===----------------------------------------------------------------------===//

void fir::IterWhileOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result, aiir::Value lb,
                             aiir::Value ub, aiir::Value step,
                             aiir::Value iterate, bool finalCountValue,
                             aiir::ValueRange iterArgs,
                             llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step, iterate});
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(getFinalValueAttrNameStr(), builder.getUnitAttr());
  }
  result.addTypes(iterate.getType());
  result.addOperands(iterArgs);
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  aiir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new aiir::Block{});
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArgument(iterate.getType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<aiir::Location>(iterArgs.size(), result.location));
  result.addAttributes(attributes);
}

aiir::ParseResult fir::IterWhileOp::parse(aiir::OpAsmParser &parser,
                                          aiir::OperationState &result) {
  auto &builder = parser.getBuilder();
  aiir::OpAsmParser::Argument inductionVariable, iterateVar;
  aiir::OpAsmParser::UnresolvedOperand lb, ub, step, iterateInput;
  if (parser.parseLParen() || parser.parseArgument(inductionVariable) ||
      parser.parseEqual())
    return aiir::failure();

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
    return aiir::failure();

  // Parse the initial iteration arguments.
  auto prependCount = false;

  // Induction variable.
  llvm::SmallVector<aiir::OpAsmParser::Argument> regionArgs;
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(iterateVar);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<aiir::Type> regionTypes;
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(regionTypes))
      return aiir::failure();
    if (regionTypes.size() == operands.size() + 2)
      prependCount = true;
    llvm::ArrayRef<aiir::Type> resTypes = regionTypes;
    resTypes = prependCount ? resTypes.drop_front(2) : resTypes;
    // Resolve input operands.
    for (auto operandType : llvm::zip(operands, resTypes))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return aiir::failure();
    if (prependCount) {
      result.addTypes(regionTypes);
    } else {
      result.addTypes(i1Type);
      result.addTypes(resTypes);
    }
  } else if (succeeded(parser.parseOptionalArrow())) {
    llvm::SmallVector<aiir::Type> typeList;
    if (parser.parseLParen() || parser.parseTypeList(typeList) ||
        parser.parseRParen())
      return aiir::failure();
    // Type list must be "(index, i1)".
    if (typeList.size() != 2 || !aiir::isa<aiir::IndexType>(typeList[0]) ||
        !typeList[1].isSignlessInteger(1))
      return aiir::failure();
    result.addTypes(typeList);
    prependCount = true;
  } else {
    result.addTypes(i1Type);
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return aiir::failure();

  llvm::SmallVector<aiir::Type> argTypes;
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
    return aiir::failure();

  fir::IterWhileOp::ensureTerminator(*body, builder, result.location);
  return aiir::success();
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
    if (!aiir::isa<aiir::IndexType>(getResult(0).getType()))
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
    return aiir::failure();
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
  return aiir::success();
}

void fir::IterWhileOp::print(aiir::OpAsmPrinter &p) {
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

llvm::SmallVector<aiir::Region *> fir::IterWhileOp::getLoopRegions() {
  return {&getRegion()};
}

aiir::BlockArgument fir::IterWhileOp::iterArgToBlockArg(aiir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

void fir::IterWhileOp::resultToSourceOps(
    llvm::SmallVectorImpl<aiir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

aiir::Value fir::IterWhileOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

std::optional<llvm::MutableArrayRef<aiir::OpOperand>>
fir::IterWhileOp::getYieldedValuesMutable() {
  auto *term = getRegion().front().getTerminator();
  return getFinalValue() ? term->getOpOperands().drop_front()
                         : term->getOpOperands();
}

//===----------------------------------------------------------------------===//
// LenParamIndexOp
//===----------------------------------------------------------------------===//

aiir::ParseResult fir::LenParamIndexOp::parse(aiir::OpAsmParser &parser,
                                              aiir::OperationState &result) {
  return parseFieldLikeOp<fir::LenType>(parser, result);
}

void fir::LenParamIndexOp::print(aiir::OpAsmPrinter &p) {
  printFieldLikeOp(p, *this);
}

void fir::LenParamIndexOp::build(aiir::OpBuilder &builder,
                                 aiir::OperationState &result,
                                 llvm::StringRef fieldName, aiir::Type recTy,
                                 aiir::ValueRange operands) {
  result.addAttribute(getFieldAttrName(), builder.getStringAttr(fieldName));
  result.addAttribute(getTypeAttrName(), aiir::TypeAttr::get(recTy));
  result.addOperands(operands);
}

llvm::SmallVector<aiir::Attribute> fir::LenParamIndexOp::getAttributes() {
  llvm::SmallVector<aiir::Attribute> attrs;
  attrs.push_back(getFieldIdAttr());
  attrs.push_back(getOnTypeAttr());
  return attrs;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static bool isSlotOrDeclaredSlot(aiir::Value val,
                                 const aiir::MemorySlot &slot) {
  if (val == slot.ptr)
    return true;
  if (auto declareOp = val.getDefiningOp<fir::DeclareOp>())
    return declareOp.getMemref() == slot.ptr;
  return false;
}

bool fir::LoadOp::loadsFrom(const aiir::MemorySlot &slot) {
  return isSlotOrDeclaredSlot(getMemref(), slot);
}

bool fir::LoadOp::storesTo(const aiir::MemorySlot &slot) { return false; }

aiir::Value fir::LoadOp::getStored(const aiir::MemorySlot &slot,
                                   aiir::OpBuilder &builder,
                                   aiir::Value reachingDef,
                                   const aiir::DataLayout &dataLayout) {
  return aiir::Value();
}

bool fir::LoadOp::canUsesBeRemoved(
    const aiir::MemorySlot &slot,
    const SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::SmallVectorImpl<aiir::OpOperand *> &newBlockingUses,
    const aiir::DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  aiir::Value blockingUse = (*blockingUses.begin())->get();
  return isSlotOrDeclaredSlot(blockingUse, slot) && getMemref() == blockingUse;
}

aiir::DeletionKind fir::LoadOp::removeBlockingUses(
    const aiir::MemorySlot &slot,
    const SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::OpBuilder &builder, aiir::Value reachingDefinition,
    const aiir::DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return aiir::DeletionKind::Delete;
}

void fir::LoadOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                        aiir::Value refVal) {
  if (!refVal) {
    aiir::emitError(result.location, "LoadOp has null argument");
    return;
  }
  auto eleTy = fir::dyn_cast_ptrEleTy(refVal.getType());
  if (!eleTy) {
    aiir::emitError(result.location, "not a memory reference type");
    return;
  }
  build(builder, result, eleTy, refVal);
}

void fir::LoadOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                        aiir::Type resTy, aiir::Value refVal) {

  if (!refVal) {
    aiir::emitError(result.location, "LoadOp has null argument");
    return;
  }
  result.addOperands(refVal);
  result.addTypes(resTy);
}

aiir::ParseResult fir::LoadOp::getElementOf(aiir::Type &ele, aiir::Type ref) {
  if ((ele = fir::dyn_cast_ptrEleTy(ref)))
    return aiir::success();
  return aiir::failure();
}

aiir::ParseResult fir::LoadOp::parse(aiir::OpAsmParser &parser,
                                     aiir::OperationState &result) {
  aiir::Type type;
  aiir::OpAsmParser::UnresolvedOperand oper;
  if (parser.parseOperand(oper) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, type, result.operands))
    return aiir::failure();
  aiir::Type eleTy;
  if (fir::LoadOp::getElementOf(eleTy, type) ||
      parser.addTypeToList(eleTy, result.types))
    return aiir::failure();
  return aiir::success();
}

void fir::LoadOp::print(aiir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getMemref());
  p.printOptionalAttrDict(getOperation()->getAttrs(), {});
  p << " : " << getMemref().getType();
}

void fir::LoadOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(aiir::MemoryEffects::Read::get(), &getMemrefMutable(),
                       aiir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// DoLoopOp
//===----------------------------------------------------------------------===//

void fir::DoLoopOp::build(aiir::OpBuilder &builder,
                          aiir::OperationState &result, aiir::Value lb,
                          aiir::Value ub, aiir::Value step, bool unordered,
                          bool finalCountValue, aiir::ValueRange iterArgs,
                          aiir::ValueRange reduceOperands,
                          llvm::ArrayRef<aiir::Attribute> reduceAttrs,
                          llvm::ArrayRef<aiir::NamedAttribute> attributes) {
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
  aiir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new aiir::Block{});
  if (iterArgs.empty() && !finalCountValue)
    fir::DoLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  bodyRegion->front().addArgument(builder.getIndexType(), result.location);
  bodyRegion->front().addArguments(
      iterArgs.getTypes(),
      llvm::SmallVector<aiir::Location>(iterArgs.size(), result.location));
  if (unordered)
    result.addAttribute(getUnorderedAttrName(result.name),
                        builder.getUnitAttr());
  if (!reduceAttrs.empty())
    result.addAttribute(getReduceAttrsAttrName(result.name),
                        builder.getArrayAttr(reduceAttrs));
  result.addAttributes(attributes);
}

aiir::ParseResult fir::DoLoopOp::parse(aiir::OpAsmParser &parser,
                                       aiir::OperationState &result) {
  auto &builder = parser.getBuilder();
  aiir::OpAsmParser::Argument inductionVariable;
  aiir::OpAsmParser::UnresolvedOperand lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual())
    return aiir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return aiir::failure();

  if (aiir::succeeded(parser.parseOptionalKeyword("unordered")))
    result.addAttribute("unordered", builder.getUnitAttr());

  // Parse the reduction arguments.
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> reduceOperands;
  llvm::SmallVector<aiir::Type> reduceArgTypes;
  if (succeeded(parser.parseOptionalKeyword("reduce"))) {
    // Parse reduction attributes and variables.
    llvm::SmallVector<ReduceAttr> attributes;
    if (failed(parser.parseCommaSeparatedList(
            aiir::AsmParser::Delimiter::Paren, [&]() {
              if (parser.parseAttribute(attributes.emplace_back()) ||
                  parser.parseArrow() ||
                  parser.parseOperand(reduceOperands.emplace_back()) ||
                  parser.parseColonType(reduceArgTypes.emplace_back()))
                return aiir::failure();
              return aiir::success();
            })))
      return aiir::failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(reduceOperands, reduceArgTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return aiir::failure();
    llvm::SmallVector<aiir::Attribute> arrayAttr(attributes.begin(),
                                                 attributes.end());
    result.addAttribute(getReduceAttrsAttrName(result.name),
                        builder.getArrayAttr(arrayAttr));
  }

  // Parse the optional initial iteration arguments.
  llvm::SmallVector<aiir::OpAsmParser::Argument> regionArgs;
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> iterOperands;
  llvm::SmallVector<aiir::Type> argTypes;
  bool prependCount = false;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, iterOperands) ||
        parser.parseArrowTypeList(result.types))
      return aiir::failure();
    if (result.types.size() == iterOperands.size() + 1)
      prependCount = true;
    // Resolve input operands.
    llvm::ArrayRef<aiir::Type> resTypes = result.types;
    for (auto operand_type : llvm::zip(
             iterOperands, prependCount ? resTypes.drop_front() : resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return aiir::failure();
  } else if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseKeyword("index"))
      return aiir::failure();
    result.types.push_back(indexType);
    prependCount = true;
  }

  // Set the operandSegmentSizes attribute
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, 1, 1, static_cast<int32_t>(reduceOperands.size()),
                           static_cast<int32_t>(iterOperands.size())}));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return aiir::failure();

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
    return aiir::failure();

  DoLoopOp::ensureTerminator(*body, builder, result.location);

  return aiir::success();
}

fir::DoLoopOp fir::getForInductionVarOwner(aiir::Value val) {
  auto ivArg = aiir::dyn_cast<aiir::BlockArgument>(val);
  if (!ivArg)
    return {};
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return aiir::dyn_cast_or_null<fir::DoLoopOp>(containingInst);
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
    return aiir::success();

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
  return aiir::success();
}

void fir::DoLoopOp::print(aiir::OpAsmPrinter &p) {
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

llvm::SmallVector<aiir::Region *> fir::DoLoopOp::getLoopRegions() {
  return {&getRegion()};
}

/// Translate a value passed as an iter_arg to the corresponding block
/// argument in the body of the loop.
aiir::BlockArgument fir::DoLoopOp::iterArgToBlockArg(aiir::Value iterArg) {
  for (auto i : llvm::enumerate(getInitArgs()))
    if (iterArg == i.value())
      return getRegion().front().getArgument(i.index() + 1);
  return {};
}

/// Translate the result vector (by index number) to the corresponding value
/// to the `fir.result` Op.
void fir::DoLoopOp::resultToSourceOps(
    llvm::SmallVectorImpl<aiir::Value> &results, unsigned resultNum) {
  auto oper = getFinalValue() ? resultNum + 1 : resultNum;
  auto *term = getRegion().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

/// Translate the block argument (by index number) to the corresponding value
/// passed as an iter_arg to the parent DoLoopOp.
aiir::Value fir::DoLoopOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= getInitArgs().size())
    return getInitArgs()[blockArgNum - 1];
  return {};
}

std::optional<llvm::MutableArrayRef<aiir::OpOperand>>
fir::DoLoopOp::getYieldedValuesMutable() {
  auto *term = getRegion().front().getTerminator();
  return getFinalValue() ? term->getOpOperands().drop_front()
                         : term->getOpOperands();
}

//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//

aiir::ParseResult fir::DTEntryOp::parse(aiir::OpAsmParser &parser,
                                        aiir::OperationState &result) {
  llvm::StringRef methodName;
  // allow `methodName` or `"methodName"`
  if (failed(parser.parseOptionalKeyword(&methodName))) {
    aiir::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr, getMethodAttrName(result.name),
                              result.attributes))
      return aiir::failure();
  } else {
    result.addAttribute(getMethodAttrName(result.name),
                        parser.getBuilder().getStringAttr(methodName));
  }
  aiir::SymbolRefAttr calleeAttr;
  if (parser.parseComma() ||
      parser.parseAttribute(calleeAttr, fir::DTEntryOp::getProcAttrNameStr(),
                            result.attributes))
    return aiir::failure();

  // Optional "deferred" keyword.
  if (succeeded(parser.parseOptionalKeyword("deferred"))) {
    result.addAttribute(fir::DTEntryOp::getDeferredAttrNameStr(),
                        parser.getBuilder().getUnitAttr());
  }
  return aiir::success();
}

void fir::DTEntryOp::print(aiir::OpAsmPrinter &p) {
  p << ' ' << getMethodAttr() << ", " << getProcAttr();
  if ((*this)->getAttr(fir::DTEntryOp::getDeferredAttrNameStr()))
    p << " deferred";
}

//===----------------------------------------------------------------------===//
// ReboxOp
//===----------------------------------------------------------------------===//

/// Get the scalar type related to a fir.box type.
/// Example: return f32 for !fir.box<!fir.heap<!fir.array<?x?xf32>>.
static aiir::Type getBoxScalarEleTy(aiir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
    return seqTy.getEleTy();
  return eleTy;
}

/// Test if \p t1 and \p t2 are compatible character types (if they can
/// represent the same type at runtime).
static bool areCompatibleCharacterTypes(aiir::Type t1, aiir::Type t2) {
  auto c1 = aiir::dyn_cast<fir::CharacterType>(t1);
  auto c2 = aiir::dyn_cast<fir::CharacterType>(t2);
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
    if (aiir::cast<fir::SliceType>(sliceVal.getType()).getRank() != inputRank)
      return emitOpError("slice operand rank must match box operand rank");
    if (auto shapeVal = getShape()) {
      if (auto shiftTy = aiir::dyn_cast<fir::ShiftType>(shapeVal.getType())) {
        if (shiftTy.getRank() != inputRank)
          return emitOpError("shape operand and input box ranks must match "
                             "when there is a slice");
      } else {
        return emitOpError("shape operand must absent or be a fir.shift "
                           "when there is a slice");
      }
    }
    if (auto sliceOp = sliceVal.getDefiningOp()) {
      auto slicedRank = aiir::cast<fir::SliceOp>(sliceOp).getOutRank();
      if (slicedRank != outRank)
        return emitOpError("result type rank and rank after applying slice "
                           "operand must match");
    }
  } else {
    // Reshaping case
    unsigned shapeRank = inputRank;
    if (auto shapeVal = getShape()) {
      auto ty = shapeVal.getType();
      if (auto shapeTy = aiir::dyn_cast<fir::ShapeType>(ty)) {
        shapeRank = shapeTy.getRank();
      } else if (auto shapeShiftTy = aiir::dyn_cast<fir::ShapeShiftType>(ty)) {
        shapeRank = shapeShiftTy.getRank();
      } else {
        auto shiftTy = aiir::cast<fir::ShiftType>(ty);
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
        aiir::isa<fir::RecordType>(inputEleTy) ||
        aiir::isa<aiir::NoneType>(outEleTy) ||
        (aiir::isa<aiir::NoneType>(inputEleTy) &&
         aiir::isa<fir::RecordType>(outEleTy)) ||
        (getSlice() && aiir::isa<fir::CharacterType>(inputEleTy)) ||
        (getSlice() && fir::isa_complex(inputEleTy) &&
         aiir::isa<aiir::FloatType>(outEleTy)) ||
        areCompatibleCharacterTypes(inputEleTy, outEleTy);
    if (!typeCanMismatch)
      return emitOpError(
          "op input and output element types must match for intrinsic types");
  }
  return aiir::success();
}

std::optional<std::int64_t> fir::ReboxOp::getViewOffset(aiir::OpResult) {
  // The address offset is zero, unless there is a slice.
  // TODO: we can handle slices that leave the base address untouched.
  if (!getSlice())
    return 0;
  return std::nullopt;
}

aiir::Speculation::Speculatability fir::ReboxOp::getSpeculatability() {
  return mayBeAbsentBox(getBox()) ? aiir::Speculation::NotSpeculatable
                                  : aiir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// ReboxAssumedRankOp
//===----------------------------------------------------------------------===//

static bool areCompatibleAssumedRankElementType(aiir::Type inputEleTy,
                                                aiir::Type outEleTy) {
  if (inputEleTy == outEleTy)
    return true;
  // Output is unlimited polymorphic -> output dynamic type is the same as input
  // type.
  if (aiir::isa<aiir::NoneType>(outEleTy))
    return true;
  // Output/Input are derived types. Assuming input extends output type, output
  // dynamic type is the output static type, unless output is polymorphic.
  if (aiir::isa<fir::RecordType>(inputEleTy) &&
      aiir::isa<fir::RecordType>(outEleTy))
    return true;
  if (areCompatibleCharacterTypes(inputEleTy, outEleTy))
    return true;
  return false;
}

llvm::LogicalResult fir::ReboxAssumedRankOp::verify() {
  aiir::Type inputType = getBox().getType();
  if (!aiir::isa<fir::BaseBoxType>(inputType) && !fir::isBoxAddress(inputType))
    return emitOpError("input must be a box or box address");
  aiir::Type inputEleTy =
      aiir::cast<fir::BaseBoxType>(fir::unwrapRefType(inputType))
          .unwrapInnerType();
  aiir::Type outEleTy =
      aiir::cast<fir::BaseBoxType>(getType()).unwrapInnerType();
  if (!areCompatibleAssumedRankElementType(inputEleTy, outEleTy))
    return emitOpError("input and output element types are incompatible");
  return aiir::success();
}

void fir::ReboxAssumedRankOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  aiir::OpOperand &inputBox = getBoxMutable();
  if (fir::isBoxAddress(inputBox.get().getType()))
    effects.emplace_back(aiir::MemoryEffects::Read::get(), &inputBox,
                         aiir::SideEffects::DefaultResource::get());
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
  return aiir::success();
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

  if (aiir::isa<fir::BoxType>(resultType)) {
    if (getShape() || !getTypeparams().empty())
      return emitOpError(
          "must not have shape or length operands if the value is a fir.box");
    return aiir::success();
  }

  // fir.record or fir.array case.
  unsigned shapeTyRank = 0;
  if (auto shapeVal = getShape()) {
    auto shapeTy = shapeVal.getType();
    if (auto s = aiir::dyn_cast<fir::ShapeType>(shapeTy))
      shapeTyRank = s.getRank();
    else
      shapeTyRank = aiir::cast<fir::ShapeShiftType>(shapeTy).getRank();
  }

  auto eleTy = resultType;
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(resultType)) {
    if (seqTy.getDimension() != shapeTyRank)
      return emitOpError(
          "shape operand must be provided and have the value rank "
          "when the value is a fir.array");
    eleTy = seqTy.getEleTy();
  } else {
    if (shapeTyRank != 0)
      return emitOpError(
          "shape operand should only be provided if the value is a fir.array");
  }

  if (auto recTy = aiir::dyn_cast<fir::RecordType>(eleTy)) {
    if (recTy.getNumLenParams() != getTypeparams().size())
      return emitOpError(
          "length parameters number must match with the value type "
          "length parameters");
  } else if (auto charTy = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
    if (getTypeparams().size() > 1)
      return emitOpError(
          "no more than one length parameter must be provided for "
          "character value");
  } else {
    if (!getTypeparams().empty())
      return emitOpError(
          "length parameters must not be provided for this value type");
  }

  return aiir::success();
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
  if (!aiir::isa<aiir::IntegerType, aiir::IndexType, fir::IntegerType>(
          op.getSelector().getType()))
    return op.emitOpError("must be an integer");
  auto cases =
      op->template getAttrOfType<aiir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumDest();
  if (count == 0)
    return op.emitOpError("must have at least one successor");
  if (op.getNumConditions() != count)
    return op.emitOpError("number of cases and targets don't match");
  if (op.targetOffsetSize() != count)
    return op.emitOpError("incorrect number of successor operand groups");
  for (decltype(count) i = 0; i != count; ++i) {
    if (!aiir::isa<aiir::IntegerAttr, aiir::UnitAttr>(cases[i]))
      return op.emitOpError("invalid case alternative");
  }
  return aiir::success();
}

static aiir::ParseResult parseIntegralSwitchTerminator(
    aiir::OpAsmParser &parser, aiir::OperationState &result,
    llvm::StringRef casesAttr, llvm::StringRef operandSegmentAttr) {
  aiir::OpAsmParser::UnresolvedOperand selector;
  aiir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return aiir::failure();

  llvm::SmallVector<aiir::Attribute> ivalues;
  llvm::SmallVector<aiir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<aiir::Value>> destArgs;
  while (true) {
    aiir::Attribute ivalue; // Integer or Unit
    aiir::Block *dest;
    llvm::SmallVector<aiir::Value> destArg;
    aiir::NamedAttrList temp;
    if (parser.parseAttribute(ivalue, "i", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return aiir::failure();
    ivalues.push_back(ivalue);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (!parser.parseOptionalRSquare())
      break;
    if (parser.parseComma())
      return aiir::failure();
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
  return aiir::success();
}

template <typename OpT>
static void printIntegralSwitchTerminator(OpT op, aiir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(op.getSelector());
  p << " : " << op.getSelector().getType() << " [";
  auto cases =
      op->template getAttrOfType<aiir::ArrayAttr>(op.getCasesAttr()).getValue();
  auto count = op.getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    auto &attr = cases[i];
    if (auto intAttr = aiir::dyn_cast_or_null<aiir::IntegerAttr>(attr))
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

aiir::ParseResult fir::SelectOp::parse(aiir::OpAsmParser &parser,
                                       aiir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectOp::print(aiir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

template <typename A, typename... AdditionalArgs>
static A getSubOperands(unsigned pos, A allArgs, aiir::DenseI32ArrayAttr ranges,
                        AdditionalArgs &&...additionalArgs) {
  unsigned start = 0;
  for (unsigned i = 0; i < pos; ++i)
    start += ranges[i];
  return allArgs.slice(start, ranges[pos],
                       std::forward<AdditionalArgs>(additionalArgs)...);
}

static aiir::MutableOperandRange
getMutableSuccessorOperands(unsigned pos, aiir::MutableOperandRange operands,
                            llvm::StringRef offsetAttr) {
  aiir::Operation *owner = operands.getOwner();
  aiir::NamedAttribute targetOffsetAttr =
      *owner->getAttrDictionary().getNamed(offsetAttr);
  return getSubOperands(
      pos, operands,
      aiir::cast<aiir::DenseI32ArrayAttr>(targetOffsetAttr.getValue()),
      aiir::MutableOperandRange::OperandSegment(pos, targetOffsetAttr));
}

std::optional<aiir::OperandRange> fir::SelectOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectOp::getCompareOperands(llvm::ArrayRef<aiir::Value>, unsigned) {
  return {};
}

aiir::SuccessorOperands fir::SelectOp::getSuccessorOperands(unsigned oper) {
  return aiir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectOp::getSuccessorOperands(llvm::ArrayRef<aiir::Value> operands,
                                    unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<aiir::ValueRange>
fir::SelectOp::getSuccessorOperands(aiir::ValueRange operands, unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

//===----------------------------------------------------------------------===//
// SelectCaseOp
//===----------------------------------------------------------------------===//

std::optional<aiir::OperandRange>
fir::SelectCaseOp::getCompareOperands(unsigned cond) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  return {getSubOperands(cond, getCompareArgs(), a)};
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectCaseOp::getCompareOperands(llvm::ArrayRef<aiir::Value> operands,
                                      unsigned cond) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

std::optional<aiir::ValueRange>
fir::SelectCaseOp::getCompareOperands(aiir::ValueRange operands,
                                      unsigned cond) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

aiir::SuccessorOperands fir::SelectCaseOp::getSuccessorOperands(unsigned oper) {
  return aiir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectCaseOp::getSuccessorOperands(llvm::ArrayRef<aiir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<aiir::ValueRange>
fir::SelectCaseOp::getSuccessorOperands(aiir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

// parser for fir.select_case Op
aiir::ParseResult fir::SelectCaseOp::parse(aiir::OpAsmParser &parser,
                                           aiir::OperationState &result) {
  aiir::OpAsmParser::UnresolvedOperand selector;
  aiir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return aiir::failure();

  llvm::SmallVector<aiir::Attribute> attrs;
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> opers;
  llvm::SmallVector<aiir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<aiir::Value>> destArgs;
  llvm::SmallVector<std::int32_t> argOffs;
  std::int32_t offSize = 0;
  while (true) {
    aiir::Attribute attr;
    aiir::Block *dest;
    llvm::SmallVector<aiir::Value> destArg;
    aiir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || isValidCaseAttr(attr) ||
        parser.parseComma())
      return aiir::failure();
    attrs.push_back(attr);
    if (aiir::dyn_cast_or_null<aiir::UnitAttr>(attr)) {
      argOffs.push_back(0);
    } else if (aiir::dyn_cast_or_null<fir::ClosedIntervalAttr>(attr)) {
      aiir::OpAsmParser::UnresolvedOperand oper1;
      aiir::OpAsmParser::UnresolvedOperand oper2;
      if (parser.parseOperand(oper1) || parser.parseComma() ||
          parser.parseOperand(oper2) || parser.parseComma())
        return aiir::failure();
      opers.push_back(oper1);
      opers.push_back(oper2);
      argOffs.push_back(2);
      offSize += 2;
    } else {
      aiir::OpAsmParser::UnresolvedOperand oper;
      if (parser.parseOperand(oper) || parser.parseComma())
        return aiir::failure();
      opers.push_back(oper);
      argOffs.push_back(1);
      ++offSize;
    }
    if (parser.parseSuccessorAndUseList(dest, destArg))
      return aiir::failure();
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (aiir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return aiir::failure();
  }
  result.addAttribute(fir::SelectCaseOp::getCasesAttr(),
                      parser.getBuilder().getArrayAttr(attrs));
  if (parser.resolveOperands(opers, type, result.operands))
    return aiir::failure();
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
  return aiir::success();
}

void fir::SelectCaseOp::print(aiir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<aiir::ArrayAttr>(getCasesAttr()).getValue();
  auto count = getNumConditions();
  for (decltype(count) i = 0; i != count; ++i) {
    if (i)
      p << ", ";
    p << cases[i] << ", ";
    if (!aiir::isa<aiir::UnitAttr>(cases[i])) {
      auto caseArgs = *getCompareOperands(i);
      p.printOperand(*caseArgs.begin());
      p << ", ";
      if (aiir::isa<fir::ClosedIntervalAttr>(cases[i])) {
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
      ->getAttrOfType<aiir::DenseI32ArrayAttr>(getCompareOffsetAttr())
      .size();
}

unsigned fir::SelectCaseOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

void fir::SelectCaseOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              aiir::Value selector,
                              llvm::ArrayRef<aiir::Attribute> compareAttrs,
                              llvm::ArrayRef<aiir::ValueRange> cmpOperands,
                              llvm::ArrayRef<aiir::Block *> destinations,
                              llvm::ArrayRef<aiir::ValueRange> destOperands,
                              llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(compareAttrs));
  llvm::SmallVector<int32_t> operOffs;
  int32_t operSize = 0;
  for (auto attr : compareAttrs) {
    if (aiir::isa<fir::ClosedIntervalAttr>(attr)) {
      operOffs.push_back(2);
      operSize += 2;
    } else if (aiir::isa<aiir::UnitAttr>(attr)) {
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
void fir::SelectCaseOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              aiir::Value selector,
                              llvm::ArrayRef<aiir::Attribute> compareAttrs,
                              llvm::ArrayRef<aiir::Value> cmpOpList,
                              llvm::ArrayRef<aiir::Block *> destinations,
                              llvm::ArrayRef<aiir::ValueRange> destOperands,
                              llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  llvm::SmallVector<aiir::ValueRange> cmpOpers;
  auto iter = cmpOpList.begin();
  for (auto &attr : compareAttrs) {
    if (aiir::isa<fir::ClosedIntervalAttr>(attr)) {
      cmpOpers.push_back(aiir::ValueRange({iter, iter + 2}));
      iter += 2;
    } else if (aiir::isa<aiir::UnitAttr>(attr)) {
      cmpOpers.push_back(aiir::ValueRange{});
    } else {
      cmpOpers.push_back(aiir::ValueRange({iter, iter + 1}));
      ++iter;
    }
  }
  build(builder, result, selector, compareAttrs, cmpOpers, destinations,
        destOperands, attributes);
}

llvm::LogicalResult fir::SelectCaseOp::verify() {
  if (!aiir::isa<aiir::IntegerType, aiir::IndexType, fir::IntegerType,
                 fir::LogicalType, fir::CharacterType>(getSelector().getType()))
    return emitOpError("must be an integer, character, or logical");
  auto cases =
      getOperation()->getAttrOfType<aiir::ArrayAttr>(getCasesAttr()).getValue();
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
    if (!(aiir::isa<fir::PointIntervalAttr>(attr) ||
          aiir::isa<fir::LowerBoundAttr>(attr) ||
          aiir::isa<fir::UpperBoundAttr>(attr) ||
          aiir::isa<fir::ClosedIntervalAttr>(attr) ||
          aiir::isa<aiir::UnitAttr>(attr)))
      return emitOpError("incorrect select case attribute type");
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// SelectRankOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::SelectRankOp::verify() {
  return verifyIntegralSwitchTerminator(*this);
}

aiir::ParseResult fir::SelectRankOp::parse(aiir::OpAsmParser &parser,
                                           aiir::OperationState &result) {
  return parseIntegralSwitchTerminator(parser, result, getCasesAttr(),
                                       getOperandSegmentSizeAttr());
}

void fir::SelectRankOp::print(aiir::OpAsmPrinter &p) {
  printIntegralSwitchTerminator(*this, p);
}

std::optional<aiir::OperandRange>
fir::SelectRankOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectRankOp::getCompareOperands(llvm::ArrayRef<aiir::Value>, unsigned) {
  return {};
}

aiir::SuccessorOperands fir::SelectRankOp::getSuccessorOperands(unsigned oper) {
  return aiir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectRankOp::getSuccessorOperands(llvm::ArrayRef<aiir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<aiir::ValueRange>
fir::SelectRankOp::getSuccessorOperands(aiir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectRankOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

//===----------------------------------------------------------------------===//
// SelectTypeOp
//===----------------------------------------------------------------------===//

std::optional<aiir::OperandRange>
fir::SelectTypeOp::getCompareOperands(unsigned) {
  return {};
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectTypeOp::getCompareOperands(llvm::ArrayRef<aiir::Value>, unsigned) {
  return {};
}

aiir::SuccessorOperands fir::SelectTypeOp::getSuccessorOperands(unsigned oper) {
  return aiir::SuccessorOperands(::getMutableSuccessorOperands(
      oper, getTargetArgsMutable(), getTargetOffsetAttr()));
}

std::optional<llvm::ArrayRef<aiir::Value>>
fir::SelectTypeOp::getSuccessorOperands(llvm::ArrayRef<aiir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

std::optional<aiir::ValueRange>
fir::SelectTypeOp::getSuccessorOperands(aiir::ValueRange operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<aiir::DenseI32ArrayAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

aiir::ParseResult fir::SelectTypeOp::parse(aiir::OpAsmParser &parser,
                                           aiir::OperationState &result) {
  aiir::OpAsmParser::UnresolvedOperand selector;
  aiir::Type type;
  if (fir::parseSelector(parser, result, selector, type))
    return aiir::failure();

  llvm::SmallVector<aiir::Attribute> attrs;
  llvm::SmallVector<aiir::Block *> dests;
  llvm::SmallVector<llvm::SmallVector<aiir::Value>> destArgs;
  while (true) {
    aiir::Attribute attr;
    aiir::Block *dest;
    llvm::SmallVector<aiir::Value> destArg;
    aiir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return aiir::failure();
    attrs.push_back(attr);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (aiir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return aiir::failure();
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
  return aiir::success();
}

unsigned fir::SelectTypeOp::targetOffsetSize() {
  return (*this)
      ->getAttrOfType<aiir::DenseI32ArrayAttr>(getTargetOffsetAttr())
      .size();
}

void fir::SelectTypeOp::print(aiir::OpAsmPrinter &p) {
  p << ' ';
  p.printOperand(getSelector());
  p << " : " << getSelector().getType() << " [";
  auto cases =
      getOperation()->getAttrOfType<aiir::ArrayAttr>(getCasesAttr()).getValue();
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
  if (!aiir::isa<fir::BaseBoxType>(getSelector().getType()))
    return emitOpError("must be a fir.class or fir.box type");
  if (auto boxType = aiir::dyn_cast<fir::BoxType>(getSelector().getType()))
    if (!aiir::isa<aiir::NoneType>(boxType.getEleTy()))
      return emitOpError("selector must be polymorphic");
  auto typeGuardAttr = getCases();
  for (unsigned idx = 0; idx < typeGuardAttr.size(); ++idx)
    if (aiir::isa<aiir::UnitAttr>(typeGuardAttr[idx]) &&
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
    if (!aiir::isa<fir::ExactTypeAttr, fir::SubclassAttr, aiir::UnitAttr>(
            typeGuardAttr[i]))
      return emitOpError("invalid type-case alternative");
  }
  return aiir::success();
}

void fir::SelectTypeOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result,
                              aiir::Value selector,
                              llvm::ArrayRef<aiir::Attribute> typeOperands,
                              llvm::ArrayRef<aiir::Block *> destinations,
                              llvm::ArrayRef<aiir::ValueRange> destOperands,
                              llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(typeOperands));
  const auto count = destinations.size();
  for (aiir::Block *dest : destinations)
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
  auto shapeTy = aiir::dyn_cast<fir::ShapeType>(getType());
  assert(shapeTy && "must be a shape type");
  if (shapeTy.getRank() != size)
    return emitOpError("shape type rank mismatch");
  return aiir::success();
}

void fir::ShapeOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                         aiir::ValueRange extents) {
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
  auto shapeTy = aiir::dyn_cast<fir::ShapeShiftType>(getType());
  assert(shapeTy && "must be a shape shift type");
  if (shapeTy.getRank() * 2 != size)
    return emitOpError("shape type rank mismatch");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::ShiftOp::verify() {
  auto size = getOrigins().size();
  auto shiftTy = aiir::dyn_cast<fir::ShiftType>(getType());
  assert(shiftTy && "must be a shift type");
  if (shiftTy.getRank() != size)
    return emitOpError("shift type rank mismatch");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void fir::SliceOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                         aiir::ValueRange trips, aiir::ValueRange path,
                         aiir::ValueRange substr) {
  const auto rank = trips.size() / 3;
  auto sliceTy = fir::SliceType::get(builder.getContext(), rank);
  build(builder, result, sliceTy, trips, path, substr);
}

/// Return the output rank of a slice op. The output rank must be between 1 and
/// the rank of the array being sliced (inclusive).
unsigned fir::SliceOp::getOutputRank(aiir::ValueRange triples) {
  unsigned rank = 0;
  if (!triples.empty()) {
    for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
      auto *op = triples[i].getDefiningOp();
      if (!aiir::isa_and_nonnull<fir::UndefOp>(op))
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
  auto sliceTy = aiir::dyn_cast<fir::SliceType>(getType());
  assert(sliceTy && "must be a slice type");
  if (sliceTy.getRank() * 3 != size)
    return emitOpError("slice type rank mismatch");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

bool fir::StoreOp::loadsFrom(const aiir::MemorySlot &slot) { return false; }

bool fir::StoreOp::storesTo(const aiir::MemorySlot &slot) {
  return isSlotOrDeclaredSlot(getMemref(), slot);
}

aiir::Value fir::StoreOp::getStored(const aiir::MemorySlot &slot,
                                    aiir::OpBuilder &builder,
                                    aiir::Value reachingDef,
                                    const aiir::DataLayout &dataLayout) {
  return getValue();
}

bool fir::StoreOp::canUsesBeRemoved(
    const aiir::MemorySlot &slot,
    const SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::SmallVectorImpl<aiir::OpOperand *> &newBlockingUses,
    const aiir::DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  aiir::Value blockingUse = (*blockingUses.begin())->get();
  return isSlotOrDeclaredSlot(blockingUse, slot) &&
         getMemref() == blockingUse && getValue() != blockingUse;
}

aiir::DeletionKind fir::StoreOp::removeBlockingUses(
    const aiir::MemorySlot &slot,
    const SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::OpBuilder &builder, aiir::Value reachingDefinition,
    const aiir::DataLayout &dataLayout) {
  return aiir::DeletionKind::Delete;
}

aiir::Type fir::StoreOp::elementType(aiir::Type refType) {
  return fir::dyn_cast_ptrEleTy(refType);
}

aiir::ParseResult fir::StoreOp::parse(aiir::OpAsmParser &parser,
                                      aiir::OperationState &result) {
  aiir::Type type;
  aiir::OpAsmParser::UnresolvedOperand oper;
  aiir::OpAsmParser::UnresolvedOperand store;
  if (parser.parseOperand(oper) || parser.parseKeyword("to") ||
      parser.parseOperand(store) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(oper, fir::StoreOp::elementType(type),
                            result.operands) ||
      parser.resolveOperand(store, type, result.operands))
    return aiir::failure();
  return aiir::success();
}

void fir::StoreOp::print(aiir::OpAsmPrinter &p) {
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
  return aiir::success();
}

void fir::StoreOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                         aiir::Value value, aiir::Value memref) {
  build(builder, result, value, memref, {}, {}, {});
}

void fir::StoreOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(aiir::MemoryEffects::Write::get(), &getMemrefMutable(),
                       aiir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getMemref().getType()}, effects);
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

aiir::ParseResult fir::PrefetchOp::parse(aiir::OpAsmParser &parser,
                                         aiir::OperationState &result) {
  aiir::OpAsmParser::UnresolvedOperand memref;
  if (parser.parseOperand(memref))
    return aiir::failure();

  if (aiir::succeeded(parser.parseLBrace())) {
    llvm::StringRef kw;
    if (parser.parseKeyword(&kw))
      return aiir::failure();

    if (kw == "read")
      result.addAttribute("rw", parser.getBuilder().getBoolAttr(false));
    else if (kw == "write")
      result.addAttribute("rw", parser.getBuilder().getUnitAttr());
    else
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected either read or write keyword");

    if (parser.parseComma())
      return aiir::failure();

    if (parser.parseKeyword(&kw))
      return aiir::failure();
    if (kw == "instruction") {
      result.addAttribute("cacheType", parser.getBuilder().getBoolAttr(false));
    } else if (kw == "data") {
      result.addAttribute("cacheType", parser.getBuilder().getUnitAttr());
    } else
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected either intruction or data keyword");

    if (parser.parseComma())
      return aiir::failure();

    if (aiir::succeeded(parser.parseKeyword("localityHint"))) {
      if (parser.parseEqual())
        return aiir::failure();
      aiir::Attribute intAttr;
      if (parser.parseAttribute(intAttr))
        return aiir::failure();
      result.addAttribute("localityHint", intAttr);
    }
    if (parser.parseRBrace())
      return aiir::failure();
  }
  aiir::Type type;
  if (parser.parseColonType(type))
    return aiir::failure();

  if (parser.resolveOperand(memref, type, result.operands))
    return aiir::failure();
  return aiir::success();
}

void fir::PrefetchOp::print(aiir::OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getMemref());
  p << " {";
  if (getRw())
    p << "write";
  else
    p << "read";
  p << ", ";
  if (getCacheType())
    p << "data";
  else
    p << "instruction";
  p << ", localityHint = ";
  p << getLocalityHint();
  p << " : " << getLocalityHintAttr().getType();
  p << "} : " << getMemref().getType();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

void fir::CopyOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                        aiir::Value source, aiir::Value destination,
                        bool noOverlap) {
  aiir::UnitAttr noOverlapAttr =
      noOverlap ? builder.getUnitAttr() : aiir::UnitAttr{};
  build(builder, result, source, destination, noOverlapAttr);
}

llvm::LogicalResult fir::CopyOp::verify() {
  aiir::Type sourceType = fir::unwrapRefType(getSource().getType());
  aiir::Type destinationType = fir::unwrapRefType(getDestination().getType());
  if (sourceType != destinationType)
    return emitOpError("source and destination must have the same value type");
  return aiir::success();
}

void fir::CopyOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(aiir::MemoryEffects::Read::get(), &getSourceMutable(),
                       aiir::SideEffects::DefaultResource::get());
  effects.emplace_back(aiir::MemoryEffects::Write::get(),
                       &getDestinationMutable(),
                       aiir::SideEffects::DefaultResource::get());
  addVolatileMemoryEffects({getDestination().getType(), getSource().getType()},
                           effects);
}

//===----------------------------------------------------------------------===//
// StringLitOp
//===----------------------------------------------------------------------===//

inline fir::CharacterType::KindTy stringLitOpGetKind(fir::StringLitOp op) {
  auto eleTy = aiir::cast<fir::SequenceType>(op.getType()).getElementType();
  return aiir::cast<fir::CharacterType>(eleTy).getFKind();
}

bool fir::StringLitOp::isWideValue() { return stringLitOpGetKind(*this) != 1; }

static aiir::NamedAttribute
mkNamedIntegerAttr(aiir::OpBuilder &builder, llvm::StringRef name, int64_t v) {
  assert(v > 0);
  return builder.getNamedAttr(
      name, builder.getIntegerAttr(builder.getIntegerType(64), v));
}

void fir::StringLitOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result,
                             fir::CharacterType inType, llvm::StringRef val,
                             std::optional<int64_t> len) {
  auto valAttr = builder.getNamedAttr(value(), builder.getStringAttr(val));
  int64_t length = len ? *len : inType.getLen();
  auto lenAttr = mkNamedIntegerAttr(builder, size(), length);
  result.addAttributes({valAttr, lenAttr});
  result.addTypes(inType);
}

template <typename C>
static aiir::ArrayAttr convertToArrayAttr(aiir::OpBuilder &builder,
                                          llvm::ArrayRef<C> xlist) {
  llvm::SmallVector<aiir::Attribute> attrs;
  auto ty = builder.getIntegerType(8 * sizeof(C));
  for (auto ch : xlist)
    attrs.push_back(builder.getIntegerAttr(ty, ch));
  return builder.getArrayAttr(attrs);
}

void fir::StringLitOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result,
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

void fir::StringLitOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result,
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

void fir::StringLitOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result,
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

aiir::ParseResult fir::StringLitOp::parse(aiir::OpAsmParser &parser,
                                          aiir::OperationState &result) {
  auto &builder = parser.getBuilder();
  aiir::Attribute val;
  aiir::NamedAttrList attrs;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseAttribute(val, "fake", attrs))
    return aiir::failure();
  if (auto v = aiir::dyn_cast<aiir::StringAttr>(val))
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::value(), v));
  else if (auto v = aiir::dyn_cast<aiir::DenseElementsAttr>(val))
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::xlist(), v));
  else if (auto v = aiir::dyn_cast<aiir::ArrayAttr>(val))
    result.attributes.push_back(
        builder.getNamedAttr(fir::StringLitOp::xlist(), v));
  else
    return parser.emitError(parser.getCurrentLocation(),
                            "found an invalid constant");
  aiir::IntegerAttr sz;
  aiir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(sz, fir::StringLitOp::size(), result.attributes) ||
      parser.parseRParen() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseColonType(type))
    return aiir::failure();
  auto charTy = aiir::dyn_cast<fir::CharacterType>(type);
  if (!charTy)
    return parser.emitError(trailingTypeLoc, "must have character type");
  type = fir::CharacterType::get(builder.getContext(), charTy.getFKind(),
                                 sz.getInt());
  if (!type || parser.addTypesToList(type, result.types))
    return aiir::failure();
  return aiir::success();
}

void fir::StringLitOp::print(aiir::OpAsmPrinter &p) {
  p << ' ' << getValue() << '(';
  p << aiir::cast<aiir::IntegerAttr>(getSize()).getValue() << ") : ";
  p.printType(getType());
}

llvm::LogicalResult fir::StringLitOp::verify() {
  if (aiir::cast<aiir::IntegerAttr>(getSize()).getValue().isNegative())
    return emitOpError("size must be non-negative");
  if (auto xl = getOperation()->getAttr(fir::StringLitOp::xlist())) {
    if (auto xList = aiir::dyn_cast<aiir::ArrayAttr>(xl)) {
      for (auto a : xList)
        if (!aiir::isa<aiir::IntegerAttr>(a))
          return emitOpError("values in initializer must be integers");
    } else if (aiir::isa<aiir::DenseElementsAttr>(xl)) {
      // do nothing
    } else {
      return emitOpError("has unexpected attribute");
    }
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// UnboxProcOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::UnboxProcOp::verify() {
  if (auto eleTy = fir::dyn_cast_ptrEleTy(getRefTuple().getType()))
    if (aiir::isa<aiir::TupleType>(eleTy))
      return aiir::success();
  return emitOpError("second output argument has bad type");
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void fir::IfOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                      aiir::Value cond, bool withElseRegion) {
  build(builder, result, {}, cond, withElseRegion);
}

void fir::IfOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                      aiir::TypeRange resultTypes, aiir::Value cond,
                      bool withElseRegion) {
  result.addOperands(cond);
  result.addTypes(resultTypes);

  aiir::Region *thenRegion = result.addRegion();
  thenRegion->push_back(new aiir::Block());
  if (resultTypes.empty())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  aiir::Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new aiir::Block());
    if (resultTypes.empty())
      IfOp::ensureTerminator(*elseRegion, builder, result.location);
  }
}

// These 3 functions copied from scf.if implementation.

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control.
void fir::IfOp::getSuccessorRegions(
    aiir::RegionBranchPoint point,
    llvm::SmallVectorImpl<aiir::RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(aiir::RegionSuccessor::parent());
    return;
  }

  // Don't consider the else region if it is empty.
  regions.push_back(aiir::RegionSuccessor(&getThenRegion()));

  // Don't consider the else region if it is empty.
  aiir::Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    regions.push_back(aiir::RegionSuccessor::parent());
  else
    regions.push_back(aiir::RegionSuccessor(elseRegion));
}

aiir::ValueRange
fir::IfOp::getSuccessorInputs(aiir::RegionSuccessor successor) {
  if (successor.isParent())
    return getOperation()->getResults();
  return aiir::ValueRange();
}

void fir::IfOp::getEntrySuccessorRegions(
    llvm::ArrayRef<aiir::Attribute> operands,
    llvm::SmallVectorImpl<aiir::RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands);
  auto boolAttr =
      aiir::dyn_cast_or_null<aiir::BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&getThenRegion());

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    else
      regions.push_back(aiir::RegionSuccessor::parent());
  }
}

void fir::IfOp::getRegionInvocationBounds(
    llvm::ArrayRef<aiir::Attribute> operands,
    llvm::SmallVectorImpl<aiir::InvocationBounds> &invocationBounds) {
  if (auto cond = aiir::dyn_cast_or_null<aiir::BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

aiir::ParseResult fir::IfOp::parse(aiir::OpAsmParser &parser,
                                   aiir::OperationState &result) {
  result.regions.reserve(2);
  aiir::Region *thenRegion = result.addRegion();
  aiir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  aiir::OpAsmParser::UnresolvedOperand cond;
  aiir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return aiir::failure();

  if (aiir::succeeded(
          parser.parseOptionalKeyword(getWeightsAttrAssemblyName()))) {
    if (parser.parseLParen())
      return aiir::failure();
    aiir::DenseI32ArrayAttr weights;
    if (parser.parseCustomAttributeWithFallback(weights, aiir::Type{}))
      return aiir::failure();
    if (weights)
      result.addAttribute(getRegionWeightsAttrName(result.name), weights);
    if (parser.parseRParen())
      return aiir::failure();
  }

  if (parser.parseOptionalArrowTypeList(result.types))
    return aiir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return aiir::failure();
  fir::IfOp::ensureTerminator(*thenRegion, parser.getBuilder(),
                              result.location);

  if (aiir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return aiir::failure();
    fir::IfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                                result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return aiir::failure();
  return aiir::success();
}

llvm::LogicalResult fir::IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");

  return aiir::success();
}

void fir::IfOp::print(aiir::OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  p << ' ' << getCondition();
  if (auto weights = getRegionWeightsAttr()) {
    p << ' ' << getWeightsAttrAssemblyName() << '(';
    p.printStrippedAttrOrType(weights);
    p << ')';
  }
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
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elideAttrs=*/{getRegionWeightsAttrName()});
}

void fir::IfOp::resultToSourceOps(llvm::SmallVectorImpl<aiir::Value> &results,
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
  auto boxType = aiir::dyn_cast_or_null<fir::BaseBoxType>(
      fir::dyn_cast_ptrEleTy(getBoxRef().getType()));
  aiir::Type boxCharType;
  if (!boxType) {
    boxCharType = aiir::dyn_cast_or_null<fir::BoxCharType>(
        fir::dyn_cast_ptrEleTy(getBoxRef().getType()));
    if (!boxCharType)
      return emitOpError("box_ref operand must have !fir.ref<!fir.box<T>> or "
                         "!fir.ref<!fir.boxchar<k>> type");
    if (getField() == fir::BoxFieldAttr::derived_type)
      return emitOpError("cannot address derived_type field of a fir.boxchar");
  }
  if (getField() != fir::BoxFieldAttr::base_addr &&
      getField() != fir::BoxFieldAttr::derived_type)
    return emitOpError("cannot address provided field");
  if (getField() == fir::BoxFieldAttr::derived_type) {
    if (!fir::boxHasAddendum(boxType))
      return emitOpError("can only address derived_type field of derived type "
                         "or unlimited polymorphic fir.box");
  }
  return aiir::success();
}

void fir::BoxOffsetOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result, aiir::Value boxRef,
                             fir::BoxFieldAttr field) {
  aiir::Type valueType =
      fir::unwrapPassByRefType(fir::unwrapRefType(boxRef.getType()));
  aiir::Type resultType = valueType;
  if (field == fir::BoxFieldAttr::base_addr)
    resultType = fir::LLVMPointerType::get(fir::ReferenceType::get(valueType));
  else if (field == fir::BoxFieldAttr::derived_type)
    resultType = fir::LLVMPointerType::get(
        fir::TypeDescType::get(fir::unwrapSequenceType(valueType)));
  build(builder, result, {resultType}, boxRef, field);
}

//===----------------------------------------------------------------------===//

aiir::ParseResult fir::isValidCaseAttr(aiir::Attribute attr) {
  if (aiir::isa<aiir::UnitAttr, fir::ClosedIntervalAttr, fir::PointIntervalAttr,
                fir::LowerBoundAttr, fir::UpperBoundAttr>(attr))
    return aiir::success();
  return aiir::failure();
}

unsigned fir::getCaseArgumentOffset(llvm::ArrayRef<aiir::Attribute> cases,
                                    unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!aiir::dyn_cast_or_null<aiir::UnitAttr>(attr)) {
      ++o;
      if (aiir::dyn_cast_or_null<fir::ClosedIntervalAttr>(attr))
        ++o;
    }
  }
  return o;
}

aiir::ParseResult
fir::parseSelector(aiir::OpAsmParser &parser, aiir::OperationState &result,
                   aiir::OpAsmParser::UnresolvedOperand &selector,
                   aiir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return aiir::failure();
  return aiir::success();
}

aiir::func::FuncOp fir::createFuncOp(aiir::Location loc, aiir::ModuleOp module,
                                     llvm::StringRef name,
                                     aiir::FunctionType type,
                                     llvm::ArrayRef<aiir::NamedAttribute> attrs,
                                     const aiir::SymbolTable *symbolTable) {
  if (symbolTable)
    if (auto f = symbolTable->lookup<aiir::func::FuncOp>(name)) {
#ifdef EXPENSIVE_CHECKS
      assert(f == module.lookupSymbol<aiir::func::FuncOp>(name) &&
             "symbolTable and module out of sync");
#endif
      return f;
    }
  if (auto f = module.lookupSymbol<aiir::func::FuncOp>(name))
    return f;
  aiir::OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPointToEnd(module.getBody());
  auto result = aiir::func::FuncOp::create(modBuilder, loc, name, type, attrs);
  result.setVisibility(aiir::SymbolTable::Visibility::Private);
  return result;
}

fir::GlobalOp fir::createGlobalOp(aiir::Location loc, aiir::ModuleOp module,
                                  llvm::StringRef name, aiir::Type type,
                                  llvm::ArrayRef<aiir::NamedAttribute> attrs,
                                  const aiir::SymbolTable *symbolTable) {
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
  aiir::OpBuilder modBuilder(module.getBodyRegion());
  auto result = fir::GlobalOp::create(modBuilder, loc, name, type, attrs);
  result.setVisibility(aiir::SymbolTable::Visibility::Private);
  return result;
}

bool fir::hasHostAssociationArgument(aiir::func::FuncOp func) {
  if (auto allArgAttrs = func.getAllArgAttrs())
    for (auto attr : allArgAttrs)
      if (auto dict = aiir::dyn_cast_or_null<aiir::DictionaryAttr>(attr))
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
valueCheckFirAttributes(aiir::Value value,
                        llvm::ArrayRef<llvm::StringRef> attributeNames,
                        bool checkAny) {
  auto testAttributeSets = [&](llvm::ArrayRef<aiir::NamedAttribute> setAttrs,
                               llvm::ArrayRef<llvm::StringRef> checkAttrs) {
    if (checkAny) {
      // Return true iff any of checkAttrs attributes is present
      // in setAttrs set.
      for (llvm::StringRef checkAttrName : checkAttrs)
        if (llvm::any_of(setAttrs, [&](aiir::NamedAttribute setAttr) {
              return setAttr.getName() == checkAttrName;
            }))
          return true;

      return false;
    }

    // Return true iff all attributes from checkAttrs are present
    // in setAttrs set.
    for (aiir::StringRef checkAttrName : checkAttrs)
      if (llvm::none_of(setAttrs, [&](aiir::NamedAttribute setAttr) {
            return setAttr.getName() == checkAttrName;
          }))
        return false;

    return true;
  };
  // If this is a fir.box that was loaded, the fir attributes will be on the
  // related fir.ref<fir.box> creation.
  if (aiir::isa<fir::BoxType>(value.getType()))
    if (auto definingOp = value.getDefiningOp())
      if (auto loadOp = aiir::dyn_cast<fir::LoadOp>(definingOp))
        value = loadOp.getMemref();
  // If this is a function argument, look in the argument attributes.
  if (auto blockArg = aiir::dyn_cast<aiir::BlockArgument>(value)) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock())
      if (auto funcOp = aiir::dyn_cast<aiir::func::FuncOp>(
              blockArg.getOwner()->getParentOp()))
        return testAttributeSets(
            aiir::cast<aiir::FunctionOpInterface>(*funcOp).getArgAttrs(
                blockArg.getArgNumber()),
            attributeNames);

    // If it is not a function argument, the attributes are unknown.
    return std::nullopt;
  }

  if (auto definingOp = value.getDefiningOp()) {
    // If this is an allocated value, look at the allocation attributes.
    if (aiir::isa<fir::AllocMemOp>(definingOp) ||
        aiir::isa<fir::AllocaOp>(definingOp))
      return testAttributeSets(definingOp->getAttrs(), attributeNames);
    // If this is an imported global, look at AddrOfOp and GlobalOp attributes.
    // Both operations are looked at because use/host associated variable (the
    // AddrOfOp) can have ASYNCHRONOUS/VOLATILE attributes even if the ultimate
    // entity (the globalOp) does not have them.
    if (auto addressOfOp = aiir::dyn_cast<fir::AddrOfOp>(definingOp)) {
      if (testAttributeSets(addressOfOp->getAttrs(), attributeNames))
        return true;
      if (auto module = definingOp->getParentOfType<aiir::ModuleOp>())
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
    aiir::Value value, llvm::ArrayRef<llvm::StringRef> attributeNames) {
  std::optional<bool> mayHaveAttr =
      valueCheckFirAttributes(value, attributeNames, /*checkAny=*/true);
  return mayHaveAttr.value_or(true);
}

bool fir::valueHasFirAttribute(aiir::Value value,
                               llvm::StringRef attributeName) {
  std::optional<bool> mayHaveAttr =
      valueCheckFirAttributes(value, {attributeName}, /*checkAny=*/false);
  return mayHaveAttr.value_or(false);
}

bool fir::anyFuncArgsHaveAttr(aiir::func::FuncOp func, llvm::StringRef attr) {
  for (unsigned i = 0, end = func.getNumArguments(); i < end; ++i)
    if (func.getArgAttr(i, attr))
      return true;
  return false;
}

std::optional<std::int64_t> fir::getIntIfConstant(aiir::Value value) {
  if (auto *definingOp = value.getDefiningOp()) {
    if (auto cst = aiir::dyn_cast<aiir::arith::ConstantOp>(definingOp))
      if (auto intAttr = aiir::dyn_cast<aiir::IntegerAttr>(cst.getValue()))
        return intAttr.getInt();
    if (auto llConstOp = aiir::dyn_cast<aiir::LLVM::ConstantOp>(definingOp))
      if (auto attr = aiir::dyn_cast<aiir::IntegerAttr>(llConstOp.getValue()))
        return attr.getValue().getSExtValue();
  }
  return {};
}

bool fir::isDummyArgument(aiir::Value v) {
  auto blockArg{aiir::dyn_cast<aiir::BlockArgument>(v)};
  if (!blockArg) {
    auto defOp = v.getDefiningOp();
    if (defOp) {
      if (auto declareOp = aiir::dyn_cast<fir::DeclareOp>(defOp))
        if (declareOp.getDummyScope())
          return true;
    }
    return false;
  }

  auto *owner{blockArg.getOwner()};
  return owner->isEntryBlock() &&
         aiir::isa<aiir::FunctionOpInterface>(owner->getParentOp());
}

aiir::Type fir::applyPathToType(aiir::Type eleTy, aiir::ValueRange path) {
  for (auto i = path.begin(), end = path.end(); eleTy && i < end;) {
    eleTy = llvm::TypeSwitch<aiir::Type, aiir::Type>(eleTy)
                .Case([&](fir::RecordType ty) {
                  if (auto *op = (*i++).getDefiningOp()) {
                    if (auto off = aiir::dyn_cast<fir::FieldIndexOp>(op))
                      return ty.getType(off.getFieldName());
                    if (auto off = aiir::dyn_cast<aiir::arith::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  }
                  return aiir::Type{};
                })
                .Case([&](fir::SequenceType ty) {
                  bool valid = true;
                  const auto rank = ty.getDimension();
                  for (std::remove_const_t<decltype(rank)> ii = 0;
                       valid && ii < rank; ++ii)
                    valid = i < end && fir::isa_integer((*i++).getType());
                  return valid ? ty.getEleTy() : aiir::Type{};
                })
                .Case([&](aiir::TupleType ty) {
                  if (auto *op = (*i++).getDefiningOp())
                    if (auto off = aiir::dyn_cast<aiir::arith::ConstantOp>(op))
                      return ty.getType(fir::toInt(off));
                  return aiir::Type{};
                })
                .Case([&](aiir::ComplexType ty) {
                  if (fir::isa_integer((*i++).getType()))
                    return ty.getElementType();
                  return aiir::Type{};
                })
                .Default([&](const auto &) { return aiir::Type{}; });
  }
  return eleTy;
}

bool fir::reboxPreservesContinuity(fir::ReboxOp rebox,
                                   bool mayHaveNonDefaultLowerBounds,
                                   bool checkWhole) {
  // If slicing is not involved, then the rebox does not affect
  // the continuity of the array.
  auto sliceArg = rebox.getSlice();
  if (!sliceArg)
    return true;

  if (auto sliceOp =
          aiir::dyn_cast_or_null<fir::SliceOp>(sliceArg.getDefiningOp()))
    return isContiguousArraySlice(sliceOp, checkWhole, rebox.getBox(),
                                  mayHaveNonDefaultLowerBounds);

  return false;
}

std::optional<int64_t> fir::getAllocaByteSize(fir::AllocaOp alloca,
                                              const aiir::DataLayout &dl,
                                              const fir::KindMapping &kindMap) {
  aiir::Type type = alloca.getInType();
  // TODO: should use the constant operands when all info is not available in
  // the type.
  if (!alloca.isDynamic())
    if (auto sizeAndAlignment =
            getTypeSizeAndAlignment(alloca.getLoc(), type, dl, kindMap))
      return sizeAndAlignment->first;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// DeclareValueOp
//===----------------------------------------------------------------------===//

static bool isLegalTypeForValueDeclare(aiir::Type type) {
  return aiir::isa<aiir::IntegerType, aiir::FloatType, aiir::ComplexType,
                   fir::LogicalType>(type);
}

llvm::LogicalResult fir::DeclareValueOp::verify() {
  if (!isLegalTypeForValueDeclare(getValue().getType()))
    return emitOpError(
        "value must be a simple scalar (integer, real, complex, or logical)");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::DeclareOp::verify() {
  auto fortranVar =
      aiir::cast<fir::FortranVariableOpInterface>(this->getOperation());
  return fortranVar.verifyDeclareLikeOpImpl(getMemref());
}

bool fir::DeclareOp::canUsesBeRemoved(
    const aiir::SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::SmallVectorImpl<aiir::OpOperand *> &newBlockingUses,
    const aiir::DataLayout &dataLayout) {
  if (!isLegalTypeForValueDeclare(fir::unwrapRefType(getType())))
    return false;
  // AIIR's mem2reg computes defining blocks only from direct users of
  // the slot pointer. Stores through fir.declare are not direct users,
  // so they are not registered as defining blocks. This causes missing
  // phi nodes at join points (e.g., loop headers). Restrict promotion
  // to the single-block case where no phi nodes are needed.
  aiir::Block *declBlock = getOperation()->getBlock();
  for (aiir::OpOperand &use : getResult().getUses()) {
    if (use.getOwner()->getBlock() != declBlock)
      return false;
    newBlockingUses.push_back(&use);
  }
  return true;
}

aiir::DeletionKind fir::DeclareOp::removeBlockingUses(
    const aiir::SmallPtrSetImpl<aiir::OpOperand *> &blockingUses,
    aiir::OpBuilder &builder) {
  return aiir::DeletionKind::Delete;
}

bool fir::DeclareOp::requiresReplacedValues() { return true; }

void fir::DeclareOp::visitReplacedValues(
    llvm::ArrayRef<std::pair<aiir::Operation *, aiir::Value>> definitions,
    aiir::OpBuilder &builder) {
  for (auto [op, value] : definitions) {
    builder.setInsertionPointAfter(op);
    fir::DeclareValueOp::create(builder, getLoc(), value, getDummyScope(),
                                getUniqNameAttr(), getFortranAttrsAttr(),
                                getDataAttrAttr(), getDummyArgNoAttr());
  }
}

//===----------------------------------------------------------------------===//
// PackArrayOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::PackArrayOp::verify() {
  aiir::Type arrayType = getArray().getType();
  if (!validTypeParams(arrayType, getTypeparams(), /*allowParamsForBox=*/true))
    return emitOpError("invalid type parameters");

  if (getInnermost() && fir::getBoxRank(arrayType) == 1)
    return emitOpError(
        "'innermost' is invalid for 1D arrays, use 'whole' instead");
  return aiir::success();
}

void fir::PackArrayOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  if (getStack())
    effects.emplace_back(
        aiir::MemoryEffects::Allocate::get(),
        aiir::SideEffects::AutomaticAllocationScopeResource::get());
  else
    effects.emplace_back(aiir::MemoryEffects::Allocate::get(),
                         aiir::SideEffects::DefaultResource::get());

  if (!getNoCopy())
    effects.emplace_back(aiir::MemoryEffects::Read::get(),
                         aiir::SideEffects::DefaultResource::get());
}

static aiir::ParseResult
parsePackArrayConstraints(aiir::OpAsmParser &parser, aiir::IntegerAttr &maxSize,
                          aiir::IntegerAttr &maxElementSize,
                          aiir::IntegerAttr &minStride) {
  aiir::OperationName opName = aiir::OperationName(
      fir::PackArrayOp::getOperationName(), parser.getContext());
  struct {
    llvm::StringRef name;
    aiir::IntegerAttr &ref;
  } attributes[] = {
      {fir::PackArrayOp::getMaxSizeAttrName(opName), maxSize},
      {fir::PackArrayOp::getMaxElementSizeAttrName(opName), maxElementSize},
      {fir::PackArrayOp::getMinStrideAttrName(opName), minStride}};

  aiir::NamedAttrList parsedAttrs;
  if (succeeded(parser.parseOptionalAttrDict(parsedAttrs))) {
    for (auto parsedAttr : parsedAttrs) {
      for (auto opAttr : attributes) {
        if (parsedAttr.getName() == opAttr.name)
          opAttr.ref = aiir::cast<aiir::IntegerAttr>(parsedAttr.getValue());
      }
    }
    return aiir::success();
  }
  return aiir::failure();
}

static void printPackArrayConstraints(aiir::OpAsmPrinter &p,
                                      fir::PackArrayOp &op,
                                      const aiir::IntegerAttr &maxSize,
                                      const aiir::IntegerAttr &maxElementSize,
                                      const aiir::IntegerAttr &minStride) {
  llvm::SmallVector<aiir::NamedAttribute> attributes;
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
  return aiir::success();
}

void fir::UnpackArrayOp::getEffects(
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::MemoryEffects::Effect>>
        &effects) {
  if (getStack())
    effects.emplace_back(
        aiir::MemoryEffects::Free::get(),
        aiir::SideEffects::AutomaticAllocationScopeResource::get());
  else
    effects.emplace_back(aiir::MemoryEffects::Free::get(),
                         aiir::SideEffects::DefaultResource::get());

  if (!getNoCopy())
    effects.emplace_back(aiir::MemoryEffects::Write::get(),
                         aiir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IsContiguousBoxOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyIsContiguousBoxOp
    : public aiir::OpRewritePattern<fir::IsContiguousBoxOp> {
  using aiir::OpRewritePattern<fir::IsContiguousBoxOp>::OpRewritePattern;
  aiir::LogicalResult
  matchAndRewrite(fir::IsContiguousBoxOp op,
                  aiir::PatternRewriter &rewriter) const override;
};
} // namespace

aiir::LogicalResult SimplifyIsContiguousBoxOp::matchAndRewrite(
    fir::IsContiguousBoxOp op, aiir::PatternRewriter &rewriter) const {
  auto boxType = aiir::cast<fir::BaseBoxType>(op.getBox().getType());
  // Nothing to do for assumed-rank arrays and !fir.box<none>.
  if (boxType.isAssumedRank() || fir::isBoxNone(boxType))
    return aiir::failure();

  if (fir::getBoxRank(boxType) == 0) {
    // Scalars are always contiguous.
    aiir::Type i1Type = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<aiir::arith::ConstantOp>(
        op, i1Type, rewriter.getIntegerAttr(i1Type, 1));
    return aiir::success();
  }

  // TODO: support more patterns, e.g. a result of fir.embox without
  // the slice is contiguous. We can add fir::isSimplyContiguous(box)
  // that walks def-use to figure it out.
  return aiir::failure();
}

void fir::IsContiguousBoxOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &patterns, aiir::AIIRContext *context) {
  patterns.add<SimplifyIsContiguousBoxOp>(context);
}

//===----------------------------------------------------------------------===//
// BoxTotalElementsOp
//===----------------------------------------------------------------------===//

namespace {
struct SimplifyBoxTotalElementsOp
    : public aiir::OpRewritePattern<fir::BoxTotalElementsOp> {
  using aiir::OpRewritePattern<fir::BoxTotalElementsOp>::OpRewritePattern;
  aiir::LogicalResult
  matchAndRewrite(fir::BoxTotalElementsOp op,
                  aiir::PatternRewriter &rewriter) const override;
};
} // namespace

aiir::LogicalResult SimplifyBoxTotalElementsOp::matchAndRewrite(
    fir::BoxTotalElementsOp op, aiir::PatternRewriter &rewriter) const {
  auto boxType = aiir::cast<fir::BaseBoxType>(op.getBox().getType());
  // Nothing to do for assumed-rank arrays and !fir.box<none>.
  if (boxType.isAssumedRank() || fir::isBoxNone(boxType))
    return aiir::failure();

  if (fir::getBoxRank(boxType) == 0) {
    // Scalar: 1 element.
    rewriter.replaceOpWithNewOp<aiir::arith::ConstantOp>(
        op, op.getType(), rewriter.getIntegerAttr(op.getType(), 1));
    return aiir::success();
  }

  // TODO: support more cases, e.g. !fir.box<!fir.array<10xi32>>.
  return aiir::failure();
}

void fir::BoxTotalElementsOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &patterns, aiir::AIIRContext *context) {
  patterns.add<SimplifyBoxTotalElementsOp>(context);
}

//===----------------------------------------------------------------------===//
// IsAssumedSizeExtentOp and AssumedSizeExtentOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldIsAssumedSizeExtentOnCtor
    : public aiir::OpRewritePattern<fir::IsAssumedSizeExtentOp> {
  using aiir::OpRewritePattern<fir::IsAssumedSizeExtentOp>::OpRewritePattern;
  aiir::LogicalResult
  matchAndRewrite(fir::IsAssumedSizeExtentOp op,
                  aiir::PatternRewriter &rewriter) const override {
    if (llvm::isa_and_nonnull<fir::AssumedSizeExtentOp>(
            op.getVal().getDefiningOp())) {
      aiir::Type i1 = rewriter.getI1Type();
      rewriter.replaceOpWithNewOp<aiir::arith::ConstantOp>(
          op, i1, rewriter.getIntegerAttr(i1, 1));
      return aiir::success();
    }
    return aiir::failure();
  }
};
} // namespace

void fir::IsAssumedSizeExtentOp::getCanonicalizationPatterns(
    aiir::RewritePatternSet &patterns, aiir::AIIRContext *context) {
  patterns.add<FoldIsAssumedSizeExtentOnCtor>(context);
}

//===----------------------------------------------------------------------===//
// LocalitySpecifierOp
//===----------------------------------------------------------------------===//

// TODO This is a copy of omp::PrivateClauseOp::verifiyRegions(). Once we find a
// solution to merge both ops into one this duplication will not be needed. See:
// https://discourse.llvm.org/t/dialect-for-data-locality-sharing-specifiers-clauses-in-openmp-openacc-and-do-concurrent/86108.
llvm::LogicalResult fir::LocalitySpecifierOp::verifyRegions() {
  aiir::Type argType = getArgType();
  auto verifyTerminator = [&](aiir::Operation *terminator,
                              bool yieldsValue) -> llvm::LogicalResult {
    if (!terminator->getBlock()->getSuccessors().empty())
      return llvm::success();

    if (!llvm::isa<fir::YieldOp>(terminator))
      return aiir::emitError(terminator->getLoc())
             << "expected exit block terminator to be an `fir.yield` op.";

    YieldOp yieldOp = llvm::cast<YieldOp>(terminator);
    aiir::TypeRange yieldedTypes = yieldOp.getResults().getTypes();

    if (!yieldsValue) {
      if (yieldedTypes.empty())
        return llvm::success();

      return aiir::emitError(terminator->getLoc())
             << "Did not expect any values to be yielded.";
    }

    if (yieldedTypes.size() == 1 && yieldedTypes.front() == argType)
      return llvm::success();

    auto error = aiir::emitError(yieldOp.getLoc())
                 << "Invalid yielded value. Expected type: " << argType
                 << ", got: ";

    if (yieldedTypes.empty())
      error << "None";
    else
      error << yieldedTypes;

    return error;
  };

  auto verifyRegion = [&](aiir::Region &region, unsigned expectedNumArgs,
                          llvm::StringRef regionName,
                          bool yieldsValue) -> llvm::LogicalResult {
    assert(!region.empty());

    if (region.getNumArguments() != expectedNumArgs)
      return aiir::emitError(region.getLoc())
             << "`" << regionName << "`: "
             << "expected " << expectedNumArgs
             << " region arguments, got: " << region.getNumArguments();

    for (aiir::Block &block : region) {
      // AIIR will verify the absence of the terminator for us.
      if (!block.mightHaveTerminator())
        continue;

      if (failed(verifyTerminator(block.getTerminator(), yieldsValue)))
        return llvm::failure();
    }

    return llvm::success();
  };

  // Ensure all of the region arguments have the same type
  for (aiir::Region *region : getRegions())
    for (aiir::Type ty : region->getArgumentTypes())
      if (ty != argType)
        return emitError() << "Region argument type mismatch: got " << ty
                           << " expected " << argType << ".";

  aiir::Region &initRegion = getInitRegion();
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

// TODO This is a copy of omp::DeclareReductionOp::verifiyRegions(). Once we
// find a solution to merge both ops into one this duplication will not be
// needed.
aiir::LogicalResult fir::DeclareReductionOp::verifyRegions() {
  if (!getAllocRegion().empty()) {
    for (YieldOp yieldOp : getAllocRegion().getOps<YieldOp>()) {
      if (yieldOp.getResults().size() != 1 ||
          yieldOp.getResults().getTypes()[0] != getType())
        return emitOpError() << "expects alloc region to yield a value "
                                "of the reduction type";
    }
  }

  if (getInitializerRegion().empty())
    return emitOpError() << "expects non-empty initializer region";
  aiir::Block &initializerEntryBlock = getInitializerRegion().front();

  if (initializerEntryBlock.getNumArguments() == 1) {
    if (!getAllocRegion().empty())
      return emitOpError() << "expects two arguments to the initializer region "
                              "when an allocation region is used";
  } else if (initializerEntryBlock.getNumArguments() == 2) {
    if (getAllocRegion().empty())
      return emitOpError() << "expects one argument to the initializer region "
                              "when no allocation region is used";
  } else {
    return emitOpError()
           << "expects one or two arguments to the initializer region";
  }

  for (aiir::Value arg : initializerEntryBlock.getArguments())
    if (arg.getType() != getType())
      return emitOpError() << "expects initializer region argument to match "
                              "the reduction type";

  for (YieldOp yieldOp : getInitializerRegion().getOps<YieldOp>()) {
    if (yieldOp.getResults().size() != 1 ||
        yieldOp.getResults().getTypes()[0] != getType())
      return emitOpError() << "expects initializer region to yield a value "
                              "of the reduction type";
  }

  if (getReductionRegion().empty())
    return emitOpError() << "expects non-empty reduction region";
  aiir::Block &reductionEntryBlock = getReductionRegion().front();
  if (reductionEntryBlock.getNumArguments() != 2 ||
      reductionEntryBlock.getArgumentTypes()[0] !=
          reductionEntryBlock.getArgumentTypes()[1] ||
      reductionEntryBlock.getArgumentTypes()[0] != getType())
    return emitOpError() << "expects reduction region with two arguments of "
                            "the reduction type";
  for (YieldOp yieldOp : getReductionRegion().getOps<YieldOp>()) {
    if (yieldOp.getResults().size() != 1 ||
        yieldOp.getResults().getTypes()[0] != getType())
      return emitOpError() << "expects reduction region to yield a value "
                              "of the reduction type";
  }

  if (!getAtomicReductionRegion().empty()) {
    aiir::Block &atomicReductionEntryBlock = getAtomicReductionRegion().front();
    if (atomicReductionEntryBlock.getNumArguments() != 2 ||
        atomicReductionEntryBlock.getArgumentTypes()[0] !=
            atomicReductionEntryBlock.getArgumentTypes()[1])
      return emitOpError() << "expects atomic reduction region with two "
                              "arguments of the same type";
  }

  if (getCleanupRegion().empty())
    return aiir::success();
  aiir::Block &cleanupEntryBlock = getCleanupRegion().front();
  if (cleanupEntryBlock.getNumArguments() != 1 ||
      cleanupEntryBlock.getArgument(0).getType() != getType())
    return emitOpError() << "expects cleanup region with one argument "
                            "of the reduction type";

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// DoConcurrentOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult fir::DoConcurrentOp::verify() {
  aiir::Block *body = getBody();

  if (body->empty())
    return emitOpError("body cannot be empty");

  if (!body->mightHaveTerminator() ||
      !aiir::isa<fir::DoConcurrentLoopOp>(body->getTerminator()))
    return emitOpError("must be terminated by 'fir.do_concurrent.loop'");

  return aiir::success();
}

//===----------------------------------------------------------------------===//
// DoConcurrentLoopOp
//===----------------------------------------------------------------------===//

static aiir::ParseResult parseSpecifierList(
    aiir::OpAsmParser &parser, aiir::OperationState &result,
    llvm::StringRef specifierKeyword, llvm::StringRef symsAttrName,
    llvm::SmallVectorImpl<aiir::OpAsmParser::Argument> &regionArgs,
    llvm::SmallVectorImpl<aiir::Type> &regionArgTypes,
    int32_t &numSpecifierOperands, bool isReduce = false) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand> specifierOperands;

  if (failed(parser.parseOptionalKeyword(specifierKeyword)))
    return aiir::success();

  std::size_t oldArgTypesSize = regionArgTypes.size();
  if (failed(parser.parseLParen()))
    return aiir::failure();

  llvm::SmallVector<bool> isByRefVec;
  llvm::SmallVector<aiir::SymbolRefAttr> spceifierSymbolVec;
  llvm::SmallVector<fir::ReduceAttr> attributes;

  if (failed(parser.parseCommaSeparatedList([&]() {
        if (isReduce)
          isByRefVec.push_back(
              parser.parseOptionalKeyword("byref").succeeded());

        if (failed(parser.parseAttribute(spceifierSymbolVec.emplace_back())))
          return aiir::failure();

        if (isReduce &&
            failed(parser.parseAttribute(attributes.emplace_back())))
          return aiir::failure();

        if (parser.parseOperand(specifierOperands.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseArgument(regionArgs.emplace_back()))
          return aiir::failure();

        return aiir::success();
      })))
    return aiir::failure();

  if (failed(parser.parseColon()))
    return aiir::failure();

  if (failed(parser.parseCommaSeparatedList([&]() {
        if (failed(parser.parseType(regionArgTypes.emplace_back())))
          return aiir::failure();

        return aiir::success();
      })))
    return aiir::failure();

  if (regionArgs.size() != regionArgTypes.size())
    return parser.emitError(parser.getNameLoc(), "mismatch in number of " +
                                                     specifierKeyword.str() +
                                                     " arg and types");

  if (failed(parser.parseRParen()))
    return aiir::failure();

  for (auto operandType :
       llvm::zip_equal(specifierOperands,
                       llvm::drop_begin(regionArgTypes, oldArgTypesSize)))
    if (parser.resolveOperand(std::get<0>(operandType),
                              std::get<1>(operandType), result.operands))
      return aiir::failure();

  if (isReduce)
    result.addAttribute(
        fir::DoConcurrentLoopOp::getReduceByrefAttrName(result.name),
        isByRefVec.empty()
            ? nullptr
            : aiir::DenseBoolArrayAttr::get(builder.getContext(), isByRefVec));

  llvm::SmallVector<aiir::Attribute> symbolAttrs(spceifierSymbolVec.begin(),
                                                 spceifierSymbolVec.end());
  result.addAttribute(symsAttrName, builder.getArrayAttr(symbolAttrs));

  if (isReduce) {
    llvm::SmallVector<aiir::Attribute> arrayAttr(attributes.begin(),
                                                 attributes.end());
    result.addAttribute(
        fir::DoConcurrentLoopOp::getReduceAttrsAttrName(result.name),
        builder.getArrayAttr(arrayAttr));
  }

  numSpecifierOperands = specifierOperands.size();

  return aiir::success();
}

aiir::ParseResult fir::DoConcurrentLoopOp::parse(aiir::OpAsmParser &parser,
                                                 aiir::OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  llvm::SmallVector<aiir::OpAsmParser::Argument, 4> regionArgs;

  if (parser.parseArgumentList(regionArgs, aiir::OpAsmParser::Delimiter::Paren))
    return aiir::failure();

  llvm::SmallVector<aiir::Type> argTypes(regionArgs.size(),
                                         builder.getIndexType());

  // Parse loop bounds.
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, regionArgs.size(),
                              aiir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return aiir::failure();

  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, regionArgs.size(),
                              aiir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return aiir::failure();

  // Parse step values.
  llvm::SmallVector<aiir::OpAsmParser::UnresolvedOperand, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, regionArgs.size(),
                              aiir::OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return aiir::failure();

  int32_t numLocalOperands = 0;
  if (failed(parseSpecifierList(parser, result, "local",
                                getLocalSymsAttrName(result.name), regionArgs,
                                argTypes, numLocalOperands)))
    return aiir::failure();

  int32_t numReduceOperands = 0;
  if (failed(parseSpecifierList(
          parser, result, "reduce", getReduceSymsAttrName(result.name),
          regionArgs, argTypes, numReduceOperands, /*isReduce=*/true)))
    return aiir::failure();

  // Set `operandSegmentSizes` attribute.
  result.addAttribute(
      DoConcurrentLoopOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lower.size()),
                                    static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(numLocalOperands),
                                    static_cast<int32_t>(numReduceOperands)}));

  // Now parse the body.
  for (auto [arg, type] : llvm::zip_equal(regionArgs, argTypes))
    arg.type = type;

  aiir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return aiir::failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return aiir::failure();

  return aiir::success();
}

void fir::DoConcurrentLoopOp::print(aiir::OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments().slice(0, getNumInductionVars())
    << ") = (" << getLowerBound() << ") to (" << getUpperBound() << ") step ("
    << getStep() << ")";

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

  if (!getReduceVars().empty()) {
    p << " reduce(";
    llvm::interleaveComma(
        llvm::zip_equal(getReduceByrefAttr().asArrayRef(), getReduceSymsAttr(),
                        getReduceAttrsAttr(), getReduceVars(),
                        getRegionReduceArgs()),
        p, [&](auto it) {
          if (std::get<0>(it))
            p << "byref ";

          p << std::get<1>(it) << " " << std::get<2>(it) << " "
            << std::get<3>(it) << " -> " << std::get<4>(it);
        });
    p << " : ";
    llvm::interleaveComma(getReduceVars(), p,
                          [&](auto it) { p << it.getType(); });
    p << ")";
  }

  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{DoConcurrentLoopOp::getOperandSegmentSizeAttr(),
                       DoConcurrentLoopOp::getLocalSymsAttrName(),
                       DoConcurrentLoopOp::getReduceSymsAttrName(),
                       DoConcurrentLoopOp::getReduceAttrsAttrName(),
                       DoConcurrentLoopOp::getReduceByrefAttrName()});
}

llvm::SmallVector<aiir::Region *> fir::DoConcurrentLoopOp::getLoopRegions() {
  return {&getRegion()};
}

llvm::LogicalResult fir::DoConcurrentLoopOp::verify() {
  aiir::Operation::operand_range lbValues = getLowerBound();
  aiir::Operation::operand_range ubValues = getUpperBound();
  aiir::Operation::operand_range stepValues = getStep();
  aiir::Operation::operand_range localVars = getLocalVars();
  aiir::Operation::operand_range reduceVars = getReduceVars();

  if (lbValues.empty())
    return emitOpError(
        "needs at least one tuple element for lowerBound, upperBound and step");

  if (lbValues.size() != ubValues.size() ||
      ubValues.size() != stepValues.size())
    return emitOpError("different number of tuple elements for lowerBound, "
                       "upperBound or step");

  // Check that the body defines the same number of block arguments as the
  // number of tuple elements in step.
  aiir::Block *body = getBody();
  unsigned numIndVarArgs =
      body->getNumArguments() - localVars.size() - reduceVars.size();

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

  return aiir::success();
}

std::optional<llvm::SmallVector<aiir::Value>>
fir::DoConcurrentLoopOp::getLoopInductionVars() {
  return llvm::SmallVector<aiir::Value>{
      getBody()->getArguments().slice(0, getLowerBound().size())};
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void fir::FIROpsDialect::registerOpExternalInterfaces() {
  // Attach default declare target interfaces to operations which can be marked
  // as declare target.
  fir::GlobalOp::attachInterface<
      aiir::omp::DeclareTargetDefaultModel<fir::GlobalOp>>(*getContext());
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
