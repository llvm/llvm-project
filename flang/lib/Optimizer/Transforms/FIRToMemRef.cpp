//===-- FIRToMemRef.cpp - Convert FIR loads and stores to MemRef ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers FIR dialect memory operations to the MemRef dialect.
// In particular it:
//
//  - Rewrites `fir.alloca` to `memref.alloca`.
//
//  - Rewrites `fir.load` / `fir.store` to `memref.load` / `memref.store`.
//
//  - Allows FIR and MemRef to coexist by introducing `fir.convert` at
//    memory-use sites. Memory operations (`memref.load`, `memref.store`,
//    `memref.reinterpret_cast`, etc.) see MemRef-typed values, while the
//    original FIR-typed values remain available for non-memory uses. For
//    example:
//
//        %fir_ref = ... : !fir.ref<!fir.array<...>>
//        %memref = fir.convert %fir_ref
//                    : !fir.ref<!fir.array<...>> -> memref<...>
//        %val = memref.load %memref[...] : memref<...>
//        fir.call @callee(%fir_ref) : (!fir.ref<!fir.array<...>>) -> ()
//
//    Here the MemRef-typed value is used for `memref.load`, while the
//    original FIR-typed value is preserved for `fir.call`.
//
//  - Computes shapes, strides, and indices as needed for slices and shifts
//    and emits `memref.reinterpret_cast` when dynamic layout is required
//    (TODO: use memref.cast instead).
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/FIRToMemRefTypeConverter.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "fir-to-memref"

using namespace mlir;

namespace fir {

#define GEN_PASS_DEF_FIRTOMEMREF
#include "flang/Optimizer/Transforms/Passes.h.inc"

static bool isMarshalLike(Operation *op) {
  auto convert = dyn_cast_if_present<fir::ConvertOp>(op);
  if (!convert)
    return false;

  bool resIsMemRef = isa<MemRefType>(convert.getType());
  bool argIsMemRef = isa<MemRefType>(convert.getValue().getType());

  assert(!(resIsMemRef && argIsMemRef) &&
         "unexpected fir.convert memref -> memref in isMarshalLike");

  return resIsMemRef || argIsMemRef;
}

using MemRefInfo = FailureOr<std::pair<Value, SmallVector<Value>>>;

static llvm::cl::opt<bool> enableFIRConvertOptimizations(
    "enable-fir-convert-opts",
    llvm::cl::desc("enable emilinating redundant fir.convert in FIR-to-MemRef"),
    llvm::cl::init(false), llvm::cl::Hidden);

class FIRToMemRef : public fir::impl::FIRToMemRefBase<FIRToMemRef> {
public:
  void runOnOperation() override;

private:
  llvm::SmallSetVector<Operation *, 32> eraseOps;

  DominanceInfo *domInfo = nullptr;

  void rewriteAlloca(fir::AllocaOp, PatternRewriter &,
                     FIRToMemRefTypeConverter &);

  void rewriteLoadOp(fir::LoadOp, PatternRewriter &,
                     FIRToMemRefTypeConverter &);

  void rewriteStoreOp(fir::StoreOp, PatternRewriter &,
                      FIRToMemRefTypeConverter &);

  MemRefInfo getMemRefInfo(Value, PatternRewriter &, FIRToMemRefTypeConverter &,
                           Operation *);

  MemRefInfo convertArrayCoorOp(Operation *memOp, fir::ArrayCoorOp,
                                PatternRewriter &, FIRToMemRefTypeConverter &);

  /// Returns true if \p coordinateOp can be lowered to an indexed memref access
  /// by convertCoordinateArrayOp. This is true when the base is a reference to
  /// a statically-shaped scalar array.
  bool isArrayIndexingCoordinateOp(fir::CoordinateOp coordinateOp,
                                   FIRToMemRefTypeConverter &) const;

  /// Lower a fir.coordinate_of that indexes into a static-extent scalar array
  /// (e.g. a struct component like `A%v(i)`) to a memref + index pair.
  MemRefInfo convertCoordinateArrayOp(Operation *memOp, fir::CoordinateOp,
                                      PatternRewriter &,
                                      FIRToMemRefTypeConverter &);

  void replaceFIRMemrefs(Value, Value, PatternRewriter &) const;

  FailureOr<Value> getFIRConvert(Operation *memOp, Operation *memref,
                                 PatternRewriter &, FIRToMemRefTypeConverter &);

  FailureOr<SmallVector<Value>> getMemrefIndices(fir::ArrayCoorOp, Operation *,
                                                 PatternRewriter &, Value,
                                                 Value) const;

  bool memrefIsOptional(Operation *) const;

  Value canonicalizeIndex(Value, PatternRewriter &) const;

  // Logical section information used by FIRToMemRef. For projected slices, the
  // descriptor still owns the physical layout, so `sliceVec` intentionally
  // stays empty while `shapeVec`/`shiftVec` remain available for index math.
  struct SliceInfo {
    SmallVector<Value> shapeVec;
    SmallVector<Value> shiftVec;
    SmallVector<Value> sliceVec;
    bool hasProjectedSlice = false;
    // Constant value of the first projected-slice field, if any.
    std::optional<std::int64_t> projectedSliceStart;
  };

  template <typename OpTy>
  void collectSliceInfoFrom(OpTy op, SliceInfo &info) const;

  void populateShapeAndShift(SmallVectorImpl<Value> &shapeVec,
                             SmallVectorImpl<Value> &shiftVec,
                             fir::ShapeShiftOp shift) const;

  void populateShift(SmallVectorImpl<Value> &vec, fir::ShiftOp shift) const;

  void populateShape(SmallVectorImpl<Value> &vec, fir::ShapeOp shape) const;

  static fir::SliceOp getSliceOp(Value sliceVal) {
    return sliceVal ? sliceVal.getDefiningOp<fir::SliceOp>() : fir::SliceOp{};
  }

  static bool hasProjectedSlice(fir::SliceOp sliceOp) {
    return sliceOp && !sliceOp.getFields().empty();
  }

  // Returns the constant first projected-slice field, if available.
  static std::optional<std::int64_t>
  getProjectedSliceStartIfConstant(fir::SliceOp sliceOp) {
    auto fields = sliceOp.getFields();
    if (fields.empty())
      return std::nullopt;
    return fir::getIntIfConstant(fields.front());
  }

  unsigned getRankFromEmbox(fir::EmboxOp embox) const {
    auto memrefType = embox.getMemref().getType();
    Type unwrappedType = fir::unwrapRefType(memrefType);
    if (auto seqType = dyn_cast<fir::SequenceType>(unwrappedType))
      return seqType.getDimension();
    return 0;
  }

  bool isCompilerGeneratedAlloca(Operation *op) const;

  void copyAttribute(Operation *from, Operation *to,
                     llvm::StringRef name) const;

  Type getBaseType(Type type, bool complexBaseTypes = false) const;

  bool memrefIsDeviceData(Operation *memref) const;

  mlir::Attribute findCudaDataAttr(Value val) const;

  Value materializeBoxAddressIfNeeded(Value basePtr, PatternRewriter &rewriter,
                                      Location loc) const;
};

void FIRToMemRef::populateShapeAndShift(SmallVectorImpl<Value> &shapeVec,
                                        SmallVectorImpl<Value> &shiftVec,
                                        fir::ShapeShiftOp shift) const {
  for (mlir::OperandRange::iterator i = shift.getPairs().begin(),
                                    endIter = shift.getPairs().end();
       i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

bool FIRToMemRef::isCompilerGeneratedAlloca(Operation *op) const {
  if (!isa<fir::AllocaOp, memref::AllocaOp>(op))
    llvm_unreachable("expected alloca op");

  return !op->getAttr("bindc_name") && !op->getAttr("uniq_name");
}

void FIRToMemRef::copyAttribute(Operation *from, Operation *to,
                                llvm::StringRef name) const {
  if (Attribute value = from->getAttr(name))
    to->setAttr(name, value);
}

Type FIRToMemRef::getBaseType(Type type, bool complexBaseTypes) const {
  if (fir::isa_fir_type(type)) {
    type = fir::getFortranElementType(type);
  } else if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    type = memrefTy.getElementType();
  }

  if (!complexBaseTypes)
    if (auto complexTy = dyn_cast<ComplexType>(type))
      type = complexTy.getElementType();
  return type;
}

bool FIRToMemRef::memrefIsDeviceData(Operation *memref) const {
  if (isa<ACC_DATA_ENTRY_OPS>(memref))
    return true;

  return cuf::hasDeviceDataAttr(memref);
}

mlir::Attribute FIRToMemRef::findCudaDataAttr(Value val) const {
  Value currentVal = val;
  llvm::SmallPtrSet<Operation *, 8> visited;

  while (currentVal) {
    Operation *defOp = currentVal.getDefiningOp();
    if (!defOp || !visited.insert(defOp).second)
      break;

    if (cuf::DataAttributeAttr cudaAttr = cuf::getDataAttr(defOp))
      return cudaAttr;

    // TODO: This is a best-effort backward walk; it is easy to miss attributes
    // as FIR evolves. Long term, it would be preferable if the necessary
    // information was carried in the type system (or otherwise made available
    // without relying on a walk-back through defining ops).
    if (auto reboxOp = dyn_cast<fir::ReboxOp>(defOp)) {
      currentVal = reboxOp.getBox();
    } else if (auto convertOp = dyn_cast<fir::ConvertOp>(defOp)) {
      currentVal = convertOp->getOperand(0);
    } else if (auto emboxOp = dyn_cast<fir::EmboxOp>(defOp)) {
      currentVal = emboxOp.getMemref();
    } else if (auto boxAddrOp = dyn_cast<fir::BoxAddrOp>(defOp)) {
      currentVal = boxAddrOp.getVal();
    } else if (auto declareOp = dyn_cast<fir::DeclareOp>(defOp)) {
      currentVal = declareOp.getMemref();
    } else {
      break;
    }
  }
  return nullptr;
}

Value FIRToMemRef::materializeBoxAddressIfNeeded(Value basePtr,
                                                 PatternRewriter &rewriter,
                                                 Location loc) const {
  if (!isa<fir::BoxType>(basePtr.getType()))
    return basePtr;

  auto boxAddrOp = fir::BoxAddrOp::create(rewriter, loc, basePtr);
  if (auto cudaAttr = findCudaDataAttr(basePtr))
    boxAddrOp->setAttr(cuf::getDataAttrName(), cudaAttr);
  return boxAddrOp.getResult();
}

void FIRToMemRef::populateShift(SmallVectorImpl<Value> &vec,
                                fir::ShiftOp shift) const {
  vec.append(shift.getOrigins().begin(), shift.getOrigins().end());
}

void FIRToMemRef::populateShape(SmallVectorImpl<Value> &vec,
                                fir::ShapeOp shape) const {
  vec.append(shape.getExtents().begin(), shape.getExtents().end());
}

template <typename OpTy>
void FIRToMemRef::collectSliceInfoFrom(OpTy op, SliceInfo &info) const {
  if constexpr (std::is_same_v<OpTy, fir::ArrayCoorOp> ||
                std::is_same_v<OpTy, fir::ReboxOp> ||
                std::is_same_v<OpTy, fir::EmboxOp>) {
    Value shapeVal = op.getShape();

    if (shapeVal) {
      Operation *shapeValOp = shapeVal.getDefiningOp();

      if (auto shapeOp = dyn_cast<fir::ShapeOp>(shapeValOp)) {
        populateShape(info.shapeVec, shapeOp);
      } else if (auto shapeShiftOp = dyn_cast<fir::ShapeShiftOp>(shapeValOp)) {
        populateShapeAndShift(info.shapeVec, info.shiftVec, shapeShiftOp);
      } else if (auto shiftOp = dyn_cast<fir::ShiftOp>(shapeValOp)) {
        populateShift(info.shiftVec, shiftOp);
      }
    }

    if (auto sliceOp = getSliceOp(op.getSlice())) {
      if (hasProjectedSlice(sliceOp)) {
        info.hasProjectedSlice = true;
        info.projectedSliceStart = getProjectedSliceStartIfConstant(sliceOp);
      }
      auto triples = sliceOp.getTriples();
      info.sliceVec.append(triples.begin(), triples.end());
    }
  }
}

void FIRToMemRef::rewriteAlloca(fir::AllocaOp firAlloca,
                                PatternRewriter &rewriter,
                                FIRToMemRefTypeConverter &typeConverter) {
  if (!typeConverter.convertibleType(firAlloca.getInType()))
    return;

  if (typeConverter.isEmptyArray(firAlloca.getType()))
    return;

  rewriter.setInsertionPointAfter(firAlloca);

  Type type = firAlloca.getType();
  MemRefType memrefTy = typeConverter.convertMemrefType(type);

  Location loc = firAlloca.getLoc();

  SmallVector<Value> sizes = firAlloca.getOperands();
  std::reverse(sizes.begin(), sizes.end());

  auto alloca = memref::AllocaOp::create(rewriter, loc, memrefTy, sizes);
  copyAttribute(firAlloca, alloca, firAlloca.getBindcNameAttrName());
  copyAttribute(firAlloca, alloca, firAlloca.getUniqNameAttrName());
  copyAttribute(firAlloca, alloca, cuf::getDataAttrName());
  copyAttribute(firAlloca, alloca, acc::getVarNameAttrName());

  auto convert = fir::ConvertOp::create(rewriter, loc, type, alloca);

  rewriter.replaceOp(firAlloca, convert);

  if (isCompilerGeneratedAlloca(alloca)) {
    for (Operation *userOp : convert->getUsers()) {
      if (auto declareOp = dyn_cast<fir::DeclareOp>(userOp)) {
        LLVM_DEBUG(llvm::dbgs()
                       << "FIRToMemRef: removing declare for compiler temp:\n";
                   declareOp->dump());
        declareOp->replaceAllUsesWith(convert);
        eraseOps.insert(userOp);
      }
    }
  }
}

bool FIRToMemRef::memrefIsOptional(Operation *op) const {
  if (auto declare = dyn_cast<fir::DeclareOp>(op)) {
    if (fir::FortranVariableOpInterface(declare).isOptional())
      return true;

    Value operand = declare.getMemref();
    Operation *operandOp = operand.getDefiningOp();
    if (operandOp && isa<fir::AbsentOp>(operandOp))
      return true;
  }

  for (mlir::Value result : op->getResults())
    for (mlir::Operation *userOp : result.getUsers())
      if (isa<fir::IsPresentOp>(userOp))
        return true;

  // TODO: If `op` is not a `fir.declare`, OPTIONAL information may still be
  // present on a related `fir.declare` reached by tracing the address/box
  // through common forwarding ops (e.g. `fir.convert`, `fir.rebox`,
  // `fir.embox`, `fir.box_addr`), then checking `declare.isOptional()`. Add the
  // search after FIR improves on it.
  return false;
}

static Value castTypeToIndexType(Value originalValue,
                                 PatternRewriter &rewriter) {
  if (originalValue.getType().isIndex())
    return originalValue;

  Type indexType = rewriter.getIndexType();
  return arith::IndexCastOp::create(rewriter, originalValue.getLoc(), indexType,
                                    originalValue);
}

static bool shouldUseBoundaryBitcast(mlir::Type fromTy, mlir::Type toTy) {
  auto isBitcastCompatibleScalarType = [](mlir::Type ty) {
    return mlir::isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>(
               ty) ||
           (mlir::isa<fir::CharacterType>(ty) &&
            mlir::cast<fir::CharacterType>(ty).getLen() ==
                fir::CharacterType::singleton());
  };
  auto getKnownScalarBitWidth = [](mlir::Type ty) -> std::optional<unsigned> {
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty))
      return intTy.getWidth();
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(ty))
      return floatTy.getWidth();
    return std::nullopt;
  };

  if (fromTy == toTy)
    return false;
  const bool fromStd = fir::isa_std_type(fromTy);
  const bool toStd = fir::isa_std_type(toTy);
  if (fromStd == toStd)
    return false;
  if (!isBitcastCompatibleScalarType(fromTy) ||
      !isBitcastCompatibleScalarType(toTy))
    return false;
  auto fromBits = getKnownScalarBitWidth(fromTy);
  auto toBits = getKnownScalarBitWidth(toTy);
  if (fromBits && toBits && *fromBits != *toBits)
    return false;
  return true;
}

static mlir::Value createTypeConversion(PatternRewriter &rewriter,
                                        mlir::Location loc, mlir::Type toTy,
                                        mlir::Value value) {
  if (shouldUseBoundaryBitcast(value.getType(), toTy))
    return fir::BitcastOp::create(rewriter, loc, toTy, value);
  return fir::ConvertOp::create(rewriter, loc, toTy, value);
}

FailureOr<SmallVector<Value>>
FIRToMemRef::getMemrefIndices(fir::ArrayCoorOp arrayCoorOp, Operation *memref,
                              PatternRewriter &rewriter, Value converted,
                              Value one) const {
  IndexType indexTy = rewriter.getIndexType();
  SmallVector<Value> indices;
  Location loc = arrayCoorOp->getLoc();
  SliceInfo sliceInfo;
  int rank = arrayCoorOp.getIndices().size();
  collectSliceInfoFrom(arrayCoorOp, sliceInfo);

  // Only collect shape/shift from fir.embox, never from fir.rebox.
  //
  // The distinction is a Fortran descriptor invariant:
  //
  //                 | fir.embox              | fir.rebox
  //   --------------|------------------------|---------------------------
  //   base_addr     | raw pointer            | pre-adjusted by (lb-1)*stride
  //   Index formula | index - lb             | index - 1
  //   Collect shift | yes                    | no
  //
  // fir.embox creates a box from a raw pointer so base_addr is not adjusted;
  // the shift must be collected so index arithmetic subtracts lb correctly.
  // fir.rebox re-boxes a live descriptor whose base_addr the runtime has
  // already adjusted downward by (lb-1)*stride; collecting the shift here
  // would subtract the lower bound a second time, giving the wrong element.
  if (auto embox = dyn_cast_or_null<fir::EmboxOp>(memref)) {
    collectSliceInfoFrom(embox, sliceInfo);
    rank = getRankFromEmbox(embox);
  }

  SmallVector<Value> &shiftVec = sliceInfo.shiftVec;
  SmallVector<Value> &sliceVec = sliceInfo.sliceVec;
  SmallVector<Value> sliceLbs, sliceStrides;
  for (size_t i = 0; i < sliceVec.size(); i += 3) {
    sliceLbs.push_back(castTypeToIndexType(sliceVec[i], rewriter));
    sliceStrides.push_back(castTypeToIndexType(sliceVec[i + 2], rewriter));
  }

  const bool isShifted = !shiftVec.empty();
  const bool isSliced = !sliceVec.empty();

  ValueRange idxs = arrayCoorOp.getIndices();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

  SmallVector<bool> filledPositions(rank, false);
  for (int i = 0; i < rank; ++i) {
    Value step = isSliced ? sliceStrides[i] : one;
    Operation *stepOp = step.getDefiningOp();
    if (stepOp && mlir::isa_and_nonnull<fir::UndefOp>(stepOp)) {
      Value shift = isShifted ? shiftVec[i] : one;
      Value sliceLb = isSliced ? sliceLbs[i] : shift;
      Value offset = arith::SubIOp::create(rewriter, loc, sliceLb, shift);
      indices.push_back(offset);
      filledPositions[i] = true;
    } else {
      indices.push_back(zero);
    }
  }

  int arrayCoorIdx = 0;
  for (int i = 0; i < rank; ++i) {
    if (filledPositions[i])
      continue;

    assert((unsigned int)arrayCoorIdx < idxs.size() &&
           "empty dimension should be eliminated\n");
    Value index = canonicalizeIndex(idxs[arrayCoorIdx], rewriter);
    Type cTy = index.getType();
    if (!llvm::isa<IndexType>(cTy)) {
      assert(cTy.isSignlessInteger() && "expected signless integer type");
      index = arith::IndexCastOp::create(rewriter, loc, indexTy, index);
    }

    Value shift = isShifted ? shiftVec[i] : one;
    Value stride = isSliced ? sliceStrides[i] : one;
    Value sliceLb = isSliced ? sliceLbs[i] : shift;

    // When the array_coor has an explicit slice with a shape_shift (i.e.
    // non-default lower bounds), the indices are in the shape_shift
    // coordinate space; subtract the lower bound (shift) to get 0-based
    // memref indices. Otherwise (the slice comes from an embox, or the
    // shape has no shift), the indices are 1-based section indices;
    // subtract 1.
    bool indicesAreFortran = isShifted && arrayCoorOp.getSlice() != nullptr;
    Value indexAdjustment =
        (isSliced && !indicesAreFortran)
            ? arith::ConstantIndexOp::create(rewriter, loc, 1)
            : shift;
    Value delta = arith::SubIOp::create(rewriter, loc, index, indexAdjustment);

    Value scaled = arith::MulIOp::create(rewriter, loc, delta, stride);

    Value offset = arith::SubIOp::create(rewriter, loc, sliceLb, shift);

    Value finalIndex = arith::AddIOp::create(rewriter, loc, scaled, offset);

    indices[i] = finalIndex;
    arrayCoorIdx++;
  }

  std::reverse(indices.begin(), indices.end());

  return indices;
}

MemRefInfo
FIRToMemRef::convertArrayCoorOp(Operation *memOp, fir::ArrayCoorOp arrayCoorOp,
                                PatternRewriter &rewriter,
                                FIRToMemRefTypeConverter &typeConverter) {
  IndexType indexTy = rewriter.getIndexType();
  Value firMemref = arrayCoorOp.getMemref();
  if (!typeConverter.convertibleMemrefType(firMemref.getType()))
    return failure();

  if (typeConverter.isEmptyArray(firMemref.getType()))
    return failure();

  Location loc = arrayCoorOp->getLoc();

  // Prefer lowering the array-coordinates computation to a memref + indices.
  // This allows erasing fir.array_coor when it is only used by load/store even
  // if the base address is a block argument (e.g. region arguments).
  Operation *memref = nullptr;
  FailureOr<Value> converted;
  if (auto blockArg = dyn_cast<BlockArgument>(firMemref)) {
    rewriter.setInsertionPoint(arrayCoorOp);
    Value basePtr = materializeBoxAddressIfNeeded(blockArg, rewriter, loc);
    Type memrefTy = typeConverter.convertMemrefType(basePtr.getType());
    converted =
        fir::ConvertOp::create(rewriter, loc, memrefTy, basePtr).getResult();
    rewriter.setInsertionPointAfter(arrayCoorOp);
  } else if ((memref = firMemref.getDefiningOp()) &&
             enableFIRConvertOptimizations && isMarshalLike(memref) &&
             !fir::isa_fir_type(firMemref.getType())) {
    converted = firMemref;
    rewriter.setInsertionPoint(arrayCoorOp);
  } else {
    Operation *arrayCoorOperation = arrayCoorOp.getOperation();
    rewriter.setInsertionPoint(arrayCoorOp);
    if (memrefIsOptional(memref)) {
      auto ifOp = arrayCoorOperation->getParentOfType<scf::IfOp>();
      if (ifOp) {
        Operation *condition = ifOp.getCondition().getDefiningOp();
        if (condition && isa<fir::IsPresentOp>(condition))
          if (condition->getOperand(0) == firMemref) {
            if (arrayCoorOperation->getParentRegion() == &ifOp.getThenRegion())
              rewriter.setInsertionPointToStart(
                  &(ifOp.getThenRegion().front()));
            else if (arrayCoorOperation->getParentRegion() ==
                     &ifOp.getElseRegion())
              rewriter.setInsertionPointToStart(
                  &(ifOp.getElseRegion().front()));
          }
      }
    }

    converted = getFIRConvert(memOp, memref, rewriter, typeConverter);
    if (failed(converted))
      return failure();

    rewriter.setInsertionPointAfter(arrayCoorOp);
  }

  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  FailureOr<SmallVector<Value>> failureOrIndices =
      getMemrefIndices(arrayCoorOp, memref, rewriter, *converted, one);
  if (failed(failureOrIndices))
    return failure();
  SmallVector<Value> indices = *failureOrIndices;

  if (converted == firMemref)
    return std::pair{*converted, indices};

  Value convertedVal = *converted;
  MemRefType memRefTy = dyn_cast<MemRefType>(convertedVal.getType());

  bool isRebox = firMemref.getDefiningOp<fir::ReboxOp>() != nullptr;
  bool isDescriptor = mlir::isa<fir::BaseBoxType>(firMemref.getType()) ||
                      firMemref.getDefiningOp<fir::BoxAddrOp>() != nullptr;

  // For complex projections, reinterpret memref<d0×...×complex<T>> as
  // memref<d0×...×2×T> and append the component index (0=re, 1=im) so that
  // each load/store touches exactly sizeof(T) bytes.
  SliceInfo sliceInfo;
  collectSliceInfoFrom(arrayCoorOp, sliceInfo);
  if (auto embox = firMemref.getDefiningOp<fir::EmboxOp>())
    collectSliceInfoFrom(embox, sliceInfo);
  else if (auto rebox = firMemref.getDefiningOp<fir::ReboxOp>())
    collectSliceInfoFrom(rebox, sliceInfo);
  auto srcTy = cast<MemRefType>((*converted).getType());
  if (sliceInfo.hasProjectedSlice) {
    if (auto complexTy = dyn_cast<mlir::ComplexType>(srcTy.getElementType())) {
      if (!sliceInfo.projectedSliceStart ||
          (*sliceInfo.projectedSliceStart != 0 &&
           *sliceInfo.projectedSliceStart != 1)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "FIRToMemRef: projected complex slice selector must be constant "
               "0 (real) or 1 (imaginary), bailing out of conversion\n");
        return failure();
      }
      auto projection = *sliceInfo.projectedSliceStart;
      SmallVector<int64_t> shape(srcTy.getShape());
      shape.push_back(2);
      Value compMemref =
          fir::ConvertOp::create(
              rewriter, loc, MemRefType::get(shape, complexTy.getElementType()),
              *converted)
              .getResult();
      indices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, projection));
      return std::pair{compMemref, indices};
    }
  }

  // Static shape does not imply contiguous layout for descriptor-backed
  // entities (e.g. boxed array sections with non-unit stride). Keep the
  // reinterpret-cast path so descriptor strides are preserved.
  if (memRefTy.hasStaticShape() && !isRebox && !isDescriptor)
    return std::pair{*converted, indices};

  unsigned rank = arrayCoorOp.getIndices().size();
  if (auto embox = firMemref.getDefiningOp<fir::EmboxOp>())
    rank = getRankFromEmbox(embox);

  SmallVector<Value> sizes;
  sizes.reserve(rank);
  SmallVector<Value> strides;
  strides.reserve(rank);

  SmallVector<Value> &shapeVec = sliceInfo.shapeVec;
  if (sliceInfo.hasProjectedSlice || shapeVec.empty()) {
    // Projected slices carry their physical layout in the descriptor. Rebuild
    // the MemRef view from box metadata instead of from slice triplets.
    auto boxElementSize =
        fir::BoxEleSizeOp::create(rewriter, loc, indexTy, firMemref);

    for (unsigned i = 0; i < rank; ++i) {
      Value dim = arith::ConstantIndexOp::create(rewriter, loc, rank - i - 1);
      auto boxDims = fir::BoxDimsOp::create(rewriter, loc, indexTy, indexTy,
                                            indexTy, firMemref, dim);

      Value extent = boxDims->getResult(1);
      sizes.push_back(castTypeToIndexType(extent, rewriter));

      Value byteStride = boxDims->getResult(2);
      Value div =
          arith::DivSIOp::create(rewriter, loc, byteStride, boxElementSize);
      strides.push_back(castTypeToIndexType(div, rewriter));
    }

  } else {
    Value oneIdx =
        arith::ConstantIndexOp::create(rewriter, arrayCoorOp->getLoc(), 1);
    for (unsigned i = rank - 1; i > 0; --i) {
      Value size = shapeVec[i];
      sizes.push_back(castTypeToIndexType(size, rewriter));

      Value stride = shapeVec[0];
      for (unsigned j = 1; j <= i - 1; ++j)
        stride = arith::MulIOp::create(rewriter, loc, shapeVec[j], stride);
      strides.push_back(castTypeToIndexType(stride, rewriter));
    }

    sizes.push_back(castTypeToIndexType(shapeVec[0], rewriter));
    strides.push_back(oneIdx);
  }

  assert(strides.size() == sizes.size() && sizes.size() == rank);

  int64_t dynamicOffset = ShapedType::kDynamic;
  SmallVector<int64_t> dynamicStrides(rank, ShapedType::kDynamic);
  auto stridedLayout = StridedLayoutAttr::get(convertedVal.getContext(),
                                              dynamicOffset, dynamicStrides);

  SmallVector<int64_t> dynamicShape(rank, ShapedType::kDynamic);
  memRefTy =
      MemRefType::get(dynamicShape, memRefTy.getElementType(), stridedLayout);

  Value offset = arith::ConstantIndexOp::create(rewriter, loc, 0);

  auto reinterpret = memref::ReinterpretCastOp::create(
      rewriter, loc, memRefTy, *converted, offset, sizes, strides);

  Value result = reinterpret->getResult(0);
  return std::pair{result, indices};
}

FailureOr<Value>
FIRToMemRef::getFIRConvert(Operation *memOp, Operation *op,
                           PatternRewriter &rewriter,
                           FIRToMemRefTypeConverter &typeConverter) {
  if (enableFIRConvertOptimizations && !op->hasOneUse() &&
      !memrefIsOptional(op)) {
    for (Operation *userOp : op->getUsers()) {
      if (auto convertOp = dyn_cast<fir::ConvertOp>(userOp)) {
        Value converted = convertOp.getResult();
        if (!isa<MemRefType>(converted.getType()))
          continue;

        if (userOp->getParentOp() == memOp->getParentOp() &&
            domInfo->dominates(userOp, memOp))
          return converted;
      }
    }
  }

  assert(op->getNumResults() == 1 && "expecting one result");

  Value basePtr = op->getResult(0);

  MemRefType memrefTy = typeConverter.convertMemrefType(basePtr.getType());
  Type baseTy = memrefTy.getElementType();

  if (fir::isa_std_type(baseTy) && memrefTy.getRank() == 0) {
    if (auto convertOp = basePtr.getDefiningOp<fir::ConvertOp>()) {
      Value input = convertOp.getOperand();
      if (auto alloca = input.getDefiningOp<memref::AllocaOp>()) {
        assert(alloca.getType() == memrefTy && "expected same types");
        if (isCompilerGeneratedAlloca(alloca))
          return alloca.getResult();
      }
    }
  }

  const Location loc = op->getLoc();

  if (isa<fir::BoxType>(basePtr.getType())) {
    Operation *baseOp = basePtr.getDefiningOp();
    basePtr = materializeBoxAddressIfNeeded(basePtr, rewriter, loc);
    memrefTy = typeConverter.convertMemrefType(basePtr.getType());

    if (baseOp) {
      auto sameBaseBoxTypes = [&](Type baseType, Type memrefType) -> bool {
        Type emboxBaseTy = getBaseType(baseType, true);
        Type emboxMemrefTy = getBaseType(memrefType, true);
        return emboxBaseTy == emboxMemrefTy;
      };

      if (auto embox = dyn_cast_or_null<fir::EmboxOp>(baseOp)) {
        // A projected slice changes the element type of the boxed view.  We
        // can only lower it here when the storage element is complex<T> and
        // the projection is the real or imaginary part (i.e. %re / %im).  For
        // such cases sizeof(complex<T>) == 2*sizeof(T), so
        // divsi(byte_stride, elesize) is always an exact integer.
        //
        // Derived-type component projections (e.g. a%x, a%y) may produce a
        // non-integer element-unit stride (e.g. sizeof(T)=24,
        // sizeof(complex<f64>)=16 -> 24/16 = 1 after truncation, which is
        // wrong).  For those, the type-restriction check below fires and we
        // bail out, leaving the ops for downstream FIR-to-LLVM lowering.
        auto isComplexComponentProjection = [&](fir::EmboxOp embox) -> bool {
          if (!hasProjectedSlice(getSliceOp(embox.getSlice())))
            return false;
          Type memTy = fir::unwrapRefType(embox.getMemref().getType());
          if (auto seqTy = dyn_cast<fir::SequenceType>(memTy))
            memTy = seqTy.getEleTy();
          return mlir::isa<mlir::ComplexType>(memTy);
        };
        bool projectedSlice = isComplexComponentProjection(embox);
        if (!projectedSlice &&
            !sameBaseBoxTypes(embox.getType(), embox.getMemref().getType())) {
          LLVM_DEBUG(llvm::dbgs()
                     << "FIRToMemRef: embox base type and memref type are not "
                        "the same, bailing out of conversion\n");
          return failure();
        }
        if (embox.getSlice() &&
            embox.getSlice().getDefiningOp<fir::SliceOp>()) {
          Type originalType = embox.getMemref().getType();
          basePtr = embox.getMemref();

          if (typeConverter.convertibleMemrefType(originalType)) {
            auto convertedMemrefTy =
                typeConverter.convertMemrefType(originalType);
            memrefTy = convertedMemrefTy;
          } else {
            return failure();
          }
        }
      }

      if (auto rebox = dyn_cast<fir::ReboxOp>(baseOp)) {
        if (!sameBaseBoxTypes(rebox.getType(), rebox.getBox().getType())) {
          LLVM_DEBUG(llvm::dbgs()
                     << "FIRToMemRef: rebox base type and box type are not the "
                        "same, bailing out of conversion\n");
          return failure();
        }
        Type originalType = rebox.getBox().getType();
        if (auto boxTy = dyn_cast<fir::BoxType>(originalType))
          originalType = boxTy.getElementType();
        if (!typeConverter.convertibleMemrefType(originalType)) {
          return failure();
        } else {
          auto convertedMemrefTy =
              typeConverter.convertMemrefType(originalType);
          memrefTy = convertedMemrefTy;
        }
      }
    }
  }

  auto convert = fir::ConvertOp::create(rewriter, loc, memrefTy, basePtr);
  return convert->getResult(0);
}

Value FIRToMemRef::canonicalizeIndex(Value index,
                                     PatternRewriter &rewriter) const {
  if (auto blockArg = dyn_cast<BlockArgument>(index))
    return index;

  Operation *op = index.getDefiningOp();

  if (auto constant = dyn_cast<arith::ConstantIntOp>(op)) {
    if (!constant.getType().isIndex()) {
      Value v = arith::ConstantIndexOp::create(rewriter, op->getLoc(),
                                               constant.value());
      return v;
    }
    return constant;
  }

  if (auto extsi = dyn_cast<arith::ExtSIOp>(op)) {
    Value operand = extsi.getOperand();
    if (auto indexCast = operand.getDefiningOp<arith::IndexCastOp>()) {
      Value v = indexCast.getOperand();
      return v;
    }
    return canonicalizeIndex(operand, rewriter);
  }

  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    Value lhs = canonicalizeIndex(add.getLhs(), rewriter);
    Value rhs = canonicalizeIndex(add.getRhs(), rewriter);
    if (lhs.getType() == rhs.getType())
      return arith::AddIOp::create(rewriter, op->getLoc(), lhs, rhs);
  }
  return index;
}

bool FIRToMemRef::isArrayIndexingCoordinateOp(
    fir::CoordinateOp coordinateOp,
    FIRToMemRefTypeConverter &typeConverter) const {
  // The base must be a reference/pointer/heap to a sequence/array type.
  Type baseType = coordinateOp.getRef().getType();
  Type unwrapped = fir::dyn_cast_ptrEleTy(baseType);
  if (!unwrapped)
    return false;

  auto seqTy = dyn_cast<fir::SequenceType>(unwrapped);
  if (!seqTy)
    return false;

  // Restrict to fully static extents — dynamic arrays would need a shape
  // operand (which coordinate_of lacks) to build a valid memref descriptor.
  if (fir::hasDynamicSize(seqTy))
    return false;

  // The element type must be a convertible scalar — no derived types.
  if (!typeConverter.convertibleMemrefType(baseType))
    return false;

  return true;
}

MemRefInfo FIRToMemRef::convertCoordinateArrayOp(
    Operation *memOp, fir::CoordinateOp coordinateOp, PatternRewriter &rewriter,
    FIRToMemRefTypeConverter &typeConverter) {
  Value firBase = coordinateOp.getRef();
  Location loc = coordinateOp->getLoc();
  IndexType indexTy = rewriter.getIndexType();

  if (typeConverter.isEmptyArray(firBase.getType()))
    return failure();

  // Convert the base ref/heap/ptr to a memref.  convertMemrefType reverses the
  // FIR column-major shape to row-major, keeping it in sync with index reversal
  // below.
  rewriter.setInsertionPoint(coordinateOp);
  FailureOr<Value> converted;
  if (isa<BlockArgument>(firBase)) {
    Type memrefTy = typeConverter.convertMemrefType(firBase.getType());
    if (!memrefTy)
      return failure();
    converted =
        fir::ConvertOp::create(rewriter, loc, memrefTy, firBase).getResult();
  } else {
    converted =
        getFIRConvert(memOp, firBase.getDefiningOp(), rewriter, typeConverter);
    if (failed(converted))
      return failure();
  }
  rewriter.setInsertionPointAfter(coordinateOp);

  // The converted memref has static shape — no reinterpret_cast needed.
  assert(cast<MemRefType>(converted->getType()).hasStaticShape() &&
         "expected static shape for coordinate_of array base");

  // Collect and normalize the 0-based coor indices
  SmallVector<Value> indices;
  for (Value v : coordinateOp.getCoor()) {
    v = canonicalizeIndex(v, rewriter);
    if (!isa<IndexType>(v.getType()))
      v = arith::IndexCastOp::create(rewriter, loc, indexTy, v);
    indices.push_back(v);
  }
  std::reverse(indices.begin(), indices.end());

  return std::pair{*converted, indices};
}

MemRefInfo FIRToMemRef::getMemRefInfo(Value firMemref,
                                      PatternRewriter &rewriter,
                                      FIRToMemRefTypeConverter &typeConverter,
                                      Operation *memOp) {
  Operation *memrefOp = firMemref.getDefiningOp();
  if (!memrefOp) {
    if (auto blockArg = dyn_cast<BlockArgument>(firMemref)) {
      rewriter.setInsertionPoint(memOp);
      Type memrefTy = typeConverter.convertMemrefType(blockArg.getType());
      if (auto mt = dyn_cast<MemRefType>(memrefTy))
        if (auto inner = llvm::dyn_cast<MemRefType>(mt.getElementType()))
          memrefTy = inner;
      Value converted = fir::ConvertOp::create(rewriter, blockArg.getLoc(),
                                               memrefTy, blockArg);
      SmallVector<Value> indices;
      return std::pair{converted, indices};
    }
    llvm_unreachable(
        "FIRToMemRef: expected defining op or block argument for FIR memref");
  }

  if (auto arrayCoorOp = dyn_cast<fir::ArrayCoorOp>(memrefOp)) {
    MemRefInfo memrefInfo =
        convertArrayCoorOp(memOp, arrayCoorOp, rewriter, typeConverter);
    if (succeeded(memrefInfo)) {
      for (auto user : memrefOp->getUsers()) {
        if (!isa<fir::LoadOp, fir::StoreOp>(user)) {
          LLVM_DEBUG(
              llvm::dbgs()
                  << "FIRToMemRef: array memref used by unsupported op:\n";
              firMemref.dump(); user->dump());
          return memrefInfo;
        }
      }
      eraseOps.insert(memrefOp);
    }
    return memrefInfo;
  }

  rewriter.setInsertionPoint(memOp);

  if (isMarshalLike(memrefOp)) {
    FailureOr<Value> converted =
        getFIRConvert(memOp, memrefOp, rewriter, typeConverter);
    if (failed(converted)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "FIRToMemRef: expected FIR memref in convert, bailing "
                        "out:\n";
                 firMemref.dump());
      return failure();
    }
    SmallVector<Value> indices;
    return std::pair{*converted, indices};
  }

  if (auto declareOp = dyn_cast<fir::DeclareOp>(memrefOp)) {
    FailureOr<Value> converted =
        getFIRConvert(memOp, declareOp, rewriter, typeConverter);
    if (failed(converted)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "FIRToMemRef: unable to create convert for scalar "
                        "memref:\n";
                 firMemref.dump());
      return failure();
    }
    SmallVector<Value> indices;
    return std::pair{*converted, indices};
  }

  if (auto coordinateOp = dyn_cast<fir::CoordinateOp>(memrefOp)) {
    // Fast path: coordinate_of used as a plain array indexer on a static-extent
    // scalar array (e.g. a struct component `A%v(i)`).
    if (isArrayIndexingCoordinateOp(coordinateOp, typeConverter)) {
      MemRefInfo memrefInfo = convertCoordinateArrayOp(memOp, coordinateOp,
                                                       rewriter, typeConverter);
      if (succeeded(memrefInfo)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "FIRToMemRef: converted coordinate_of array indexer\n");
        for (auto user : memrefOp->getUsers()) {
          if (!isa<fir::LoadOp, fir::StoreOp>(user)) {
            LLVM_DEBUG(
                llvm::dbgs()
                    << "FIRToMemRef: coordinate_of used by non-load/store, "
                       "skipping erase\n";
                firMemref.dump(); user->dump());
            return memrefInfo;
          }
        }
        eraseOps.insert(memrefOp);
        return memrefInfo;
      }
    }

    // Fallback: struct field access or dynamic array — produce a rank-0 scalar
    // memref from the leaf reference.
    FailureOr<Value> converted =
        getFIRConvert(memOp, coordinateOp, rewriter, typeConverter);
    if (failed(converted)) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "FIRToMemRef: unable to create convert for derived-type "
                 "memref:\n";
          firMemref.dump());
      return failure();
    }
    SmallVector<Value> indices;
    return std::pair{*converted, indices};
  }

  if (auto convertOp = dyn_cast<fir::ConvertOp>(memrefOp)) {
    Type fromTy = convertOp->getOperand(0).getType();
    Type toTy = firMemref.getType();
    if (isa<fir::ReferenceType>(fromTy) && isa<fir::ReferenceType>(toTy)) {
      FailureOr<Value> converted =
          getFIRConvert(memOp, convertOp, rewriter, typeConverter);
      if (failed(converted)) {
        LLVM_DEBUG(
            llvm::dbgs()
                << "FIRToMemRef: unable to create convert for conversion "
                   "op:\n";
            firMemref.dump());
        return failure();
      }
      SmallVector<Value> indices;
      return std::pair{*converted, indices};
    }
  }

  if (auto boxAddrOp = dyn_cast<fir::BoxAddrOp>(memrefOp)) {
    FailureOr<Value> converted =
        getFIRConvert(memOp, boxAddrOp, rewriter, typeConverter);
    if (failed(converted)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "FIRToMemRef: unable to create convert for box_addr "
                        "op:\n";
                 firMemref.dump());
      return failure();
    }
    SmallVector<Value> indices;
    return std::pair{*converted, indices};
  }

  if (memrefIsDeviceData(memrefOp)) {
    FailureOr<Value> converted =
        getFIRConvert(memOp, memrefOp, rewriter, typeConverter);
    if (failed(converted))
      return failure();
    SmallVector<Value> indices;
    return std::pair{*converted, indices};
  }

  LLVM_DEBUG(llvm::dbgs()
                 << "FIRToMemRef: unable to create convert for memref value:\n";
             firMemref.dump());

  return failure();
}

void FIRToMemRef::replaceFIRMemrefs(Value firMemref, Value converted,
                                    PatternRewriter &rewriter) const {
  Operation *op = firMemref.getDefiningOp();
  if (op && (isa<fir::ArrayCoorOp>(op) || isMarshalLike(op)))
    return;

  SmallPtrSet<Operation *, 4> worklist;
  for (auto user : firMemref.getUsers()) {
    if (isMarshalLike(user) || isa<fir::LoadOp, fir::StoreOp>(user))
      continue;
    if (!domInfo->dominates(converted, user))
      continue;
    if (!(isa<omp::AtomicCaptureOp>(user->getParentOp()) ||
          isa<acc::AtomicCaptureOp>(user->getParentOp())))
      worklist.insert(user);
  }

  Type ty = firMemref.getType();

  for (auto op : worklist) {
    rewriter.setInsertionPoint(op);
    Location loc = op->getLoc();
    Value replaceConvert = fir::ConvertOp::create(rewriter, loc, ty, converted);
    op->replaceUsesOfWith(firMemref, replaceConvert);
  }

  worklist.clear();

  for (auto user : firMemref.getUsers()) {
    if (isMarshalLike(user) || isa<fir::LoadOp, fir::StoreOp>(user))
      continue;
    if (isa<omp::AtomicCaptureOp>(user->getParentOp()) ||
        isa<acc::AtomicCaptureOp>(user->getParentOp()))
      if (domInfo->dominates(converted, user))
        worklist.insert(user);
  }

  if (worklist.empty())
    return;

  while (!worklist.empty()) {
    Operation *parentOp = (*worklist.begin())->getParentOp();

    Value replaceConvert;
    SmallVector<Operation *> erase;
    for (auto op : worklist) {
      if (op->getParentOp() != parentOp)
        continue;
      if (!replaceConvert) {
        rewriter.setInsertionPoint(parentOp);
        replaceConvert =
            fir::ConvertOp::create(rewriter, op->getLoc(), ty, converted);
      }
      op->replaceUsesOfWith(firMemref, replaceConvert);
      erase.push_back(op);
    }

    for (auto op : erase)
      worklist.erase(op);
  }
}

void FIRToMemRef::rewriteLoadOp(fir::LoadOp load, PatternRewriter &rewriter,
                                FIRToMemRefTypeConverter &typeConverter) {
  Value firMemref = load.getMemref();
  if (!typeConverter.convertibleType(firMemref.getType()))
    return;

  LLVM_DEBUG(llvm::dbgs() << "FIRToMemRef: attempting to convert FIR load:\n";
             load.dump(); firMemref.dump());

  MemRefInfo memrefInfo =
      getMemRefInfo(firMemref, rewriter, typeConverter, load.getOperation());
  if (failed(memrefInfo))
    return;

  Type originalType = load.getResult().getType();
  Value converted = memrefInfo->first;
  SmallVector<Value> indices = memrefInfo->second;

  LLVM_DEBUG(llvm::dbgs()
                 << "FIRToMemRef: convert for FIR load created successfully:\n";
             converted.dump());

  rewriter.setInsertionPointAfter(load);

  Attribute attr = (load.getOperation())->getAttr("tbaa");
  memref::LoadOp loadOp =
      rewriter.replaceOpWithNewOp<memref::LoadOp>(load, converted, indices);
  if (attr)
    loadOp.getOperation()->setAttr("tbaa", attr);

  LLVM_DEBUG(llvm::dbgs() << "FIRToMemRef: new memref.load op:\n";
             loadOp.dump(); assert(succeeded(verify(loadOp))));

  if (loadOp.getType() != originalType) {
    Value castVal =
        createTypeConversion(rewriter, loadOp.getLoc(), originalType, loadOp);
    loadOp.getResult().replaceAllUsesExcept(castVal, castVal.getDefiningOp());
  }

  if (!isa<fir::LogicalType>(originalType))
    replaceFIRMemrefs(firMemref, converted, rewriter);
}

void FIRToMemRef::rewriteStoreOp(fir::StoreOp store, PatternRewriter &rewriter,
                                 FIRToMemRefTypeConverter &typeConverter) {
  Value firMemref = store.getMemref();

  if (!typeConverter.convertibleType(firMemref.getType()))
    return;

  LLVM_DEBUG(llvm::dbgs() << "FIRToMemRef: attempting to convert FIR store:\n";
             store.dump(); firMemref.dump());

  MemRefInfo memrefInfo =
      getMemRefInfo(firMemref, rewriter, typeConverter, store.getOperation());
  if (failed(memrefInfo))
    return;

  Value converted = memrefInfo->first;
  SmallVector<Value> indices = memrefInfo->second;
  LLVM_DEBUG(
      llvm::dbgs()
          << "FIRToMemRef: convert for FIR store created successfully:\n";
      converted.dump());

  Value value = store.getValue();
  rewriter.setInsertionPointAfter(store);

  Type convertedType = typeConverter.convertType(value.getType());
  if (convertedType != value.getType())
    value =
        createTypeConversion(rewriter, store.getLoc(), convertedType, value);

  Attribute attr = store.getOperation()->getAttr("tbaa");
  memref::StoreOp storeOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
      store, value, converted, indices);
  if (attr)
    storeOp.getOperation()->setAttr("tbaa", attr);

  LLVM_DEBUG(llvm::dbgs() << "FIRToMemRef: new memref.store op:\n";
             storeOp.dump(); assert(succeeded(verify(storeOp))));

  bool isLogicalRef = false;
  if (fir::ReferenceType refTy =
          llvm::dyn_cast<fir::ReferenceType>(firMemref.getType()))
    isLogicalRef = llvm::isa<fir::LogicalType>(refTy.getEleTy());
  if (!isLogicalRef)
    replaceFIRMemrefs(firMemref, converted, rewriter);
}

// Lower operand and result type of FIR logical operation to get rid
// of bitcast after loads from memref and before store to memref
// storing arrays of logicals as integers.
template <typename Op>
static void rewriteLogicalOperation(Op op, PatternRewriter &rewriter,
                                    FIRToMemRefTypeConverter &typeConverter) {
  mlir::Type oldType = op.getResult().getType();
  mlir::Type convertedTy = typeConverter.convertType(oldType);
  if (convertedTy == oldType)
    return;
  rewriter.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  // Inserted bitcast from/to logical will be folded with the one created
  // around load/store.
  mlir::Value lhs =
      createTypeConversion(rewriter, loc, convertedTy, op.getLhs());
  mlir::Value rhs =
      createTypeConversion(rewriter, loc, convertedTy, op.getRhs());
  auto newOp = Op::create(rewriter, loc, convertedTy, lhs, rhs);
  mlir::Value result = createTypeConversion(rewriter, loc, oldType, newOp);
  rewriter.replaceOp(op, result);
}

void FIRToMemRef::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Enter FIRToMemRef()\n");

  func::FuncOp op = getOperation();
  MLIRContext *context = op.getContext();
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  FIRToMemRefTypeConverter typeConverter(mod);

  typeConverter.setConvertComplexTypes(true);

  PatternRewriter rewriter(context);
  domInfo = new DominanceInfo(op);

  op.walk([&](fir::AllocaOp alloca) {
    rewriteAlloca(alloca, rewriter, typeConverter);
  });

  op.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<fir::LoadOp>([&](auto loadOp) {
          rewriteLoadOp(loadOp, rewriter, typeConverter);
        })
        .Case<fir::StoreOp>([&](auto storeOp) {
          rewriteStoreOp(storeOp, rewriter, typeConverter);
        })
        .Case<fir::LogicalAndOp, fir::LogicalOrOp, fir::EqvOp, fir::NeqvOp>(
            [&](auto logicalOp) {
              rewriteLogicalOperation(logicalOp, rewriter, typeConverter);
            })
        .Default([](Operation *) {});
  });

  for (auto eraseOp : eraseOps)
    rewriter.eraseOp(eraseOp);
  eraseOps.clear();

  if (domInfo)
    delete domInfo;

  LLVM_DEBUG(llvm::dbgs() << "After FIRToMemRef()\n"; op.dump();
             llvm::dbgs() << "Exit FIRToMemRef()\n";);
}

} // namespace fir
