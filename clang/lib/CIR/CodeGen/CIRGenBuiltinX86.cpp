//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit x86/x86_64 Builtin calls as CIR or a function
// call to be later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace clang;
using namespace clang::CIRGen;

// OG has unordered comparison as a form of optimization in addition to
// ordered comparison, while CIR doesn't.
//
// This means that we can't encode the comparison code of UGT (unordered
// greater than), at least not at the CIR level.
//
// The boolean shouldInvert compensates for this.
// For example: to get to the comparison code UGT, we pass in
// emitVectorFCmp (OLE, shouldInvert = true) since OLE is the inverse of UGT.

// There are several ways to support this otherwise:
// - register extra CmpOpKind for unordered comparison types and build the
// translation code for
//    to go from CIR -> LLVM dialect. Notice we get this naturally with
//    shouldInvert, benefiting from existing infrastructure, albeit having to
//    generate an extra `not` at CIR).
// - Just add extra comparison code to a new VecCmpOpKind instead of
// cluttering CmpOpKind.
// - Add a boolean in VecCmpOp to indicate if it's doing unordered or ordered
// comparison
// - Just emit the intrinsics call instead of calling this helper, see how the
// LLVM lowering handles this.
static mlir::Value emitVectorFCmp(CIRGenBuilderTy &builder,
                                  llvm::SmallVector<mlir::Value> &ops,
                                  mlir::Location loc, cir::CmpOpKind pred,
                                  bool shouldInvert) {
  assert(!cir::MissingFeatures::cgFPOptionsRAII());
  // TODO(cir): Add isSignaling boolean once emitConstrainedFPCall implemented
  assert(!cir::MissingFeatures::emitConstrainedFPCall());
  mlir::Value cmp = builder.createVecCompare(loc, pred, ops[0], ops[1]);
  mlir::Value bitCast = builder.createBitcast(
      shouldInvert ? builder.createNot(cmp) : cmp, ops[0].getType());
  return bitCast;
}

static mlir::Value getMaskVecValue(CIRGenBuilderTy &builder, mlir::Location loc,
                                   mlir::Value mask, unsigned numElems) {
  auto maskTy = cir::VectorType::get(
      builder.getSIntNTy(1), cast<cir::IntType>(mask.getType()).getWidth());
  mlir::Value maskVec = builder.createBitcast(mask, maskTy);

  // If we have less than 8 elements, then the starting mask was an i8 and
  // we need to extract down to the right number of elements.
  if (numElems < 8) {
    SmallVector<mlir::Attribute, 4> indices;
    mlir::Type i32Ty = builder.getSInt32Ty();
    for (auto i : llvm::seq<unsigned>(0, numElems))
      indices.push_back(cir::IntAttr::get(i32Ty, i));

    maskVec = builder.createVecShuffle(loc, maskVec, maskVec, indices);
  }
  return maskVec;
}

// Builds the VecShuffleOp for pshuflw and pshufhw x86 builtins.
//
// The vector is split into lanes of 8 word elements (16 bits). The lower or
// upper half of each lane, controlled by `isLow`, is shuffled in the following
// way: The immediate is truncated to 8 bits, separated into 4 2-bit fields. The
// i-th field's value represents the resulting index of the i-th element in the
// half lane after shuffling. The other half of the lane remains unchanged.
static cir::VecShuffleOp emitPshufWord(CIRGenBuilderTy &builder,
                                       const mlir::Value vec,
                                       const mlir::Value immediate,
                                       const mlir::Location loc,
                                       const bool isLow) {
  uint32_t imm = CIRGenFunction::getZExtIntValueFromConstOp(immediate);

  auto vecTy = cast<cir::VectorType>(vec.getType());
  unsigned numElts = vecTy.getSize();

  unsigned firstHalfStart = isLow ? 0 : 4;
  unsigned secondHalfStart = 4 - firstHalfStart;

  // Splat the 8-bits of immediate 4 times to help the loop wrap around.
  imm = (imm & 0xff) * 0x01010101;

  int64_t indices[32];
  for (unsigned l = 0; l != numElts; l += 8) {
    for (unsigned i = firstHalfStart; i != firstHalfStart + 4; ++i) {
      indices[l + i] = l + (imm & 3) + firstHalfStart;
      imm >>= 2;
    }
    for (unsigned i = secondHalfStart; i != secondHalfStart + 4; ++i)
      indices[l + i] = l + i;
  }

  return builder.createVecShuffle(loc, vec, ArrayRef(indices, numElts));
}

// Builds the shuffle mask for pshufd and shufpd/shufps x86 builtins.
// The shuffle mask is written to outIndices.
static void
computeFullLaneShuffleMask(CIRGenFunction &cgf, const mlir::Value vec,
                           uint32_t imm, const bool isShufP,
                           llvm::SmallVectorImpl<int64_t> &outIndices) {
  auto vecTy = cast<cir::VectorType>(vec.getType());
  unsigned numElts = vecTy.getSize();
  unsigned numLanes = cgf.cgm.getDataLayout().getTypeSizeInBits(vecTy) / 128;
  unsigned numLaneElts = numElts / numLanes;

  // Splat the 8-bits of immediate 4 times to help the loop wrap around.
  imm = (imm & 0xff) * 0x01010101;

  for (unsigned l = 0; l != numElts; l += numLaneElts) {
    for (unsigned i = 0; i != numLaneElts; ++i) {
      uint32_t idx = imm % numLaneElts;
      imm /= numLaneElts;
      if (isShufP && i >= (numLaneElts / 2))
        idx += numElts;
      outIndices[l + i] = l + idx;
    }
  }

  outIndices.resize(numElts);
}

static mlir::Value emitPrefetch(CIRGenFunction &cgf, unsigned builtinID,
                                const CallExpr *e,
                                const SmallVector<mlir::Value> &ops) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location location = cgf.getLoc(e->getExprLoc());
  mlir::Type voidTy = builder.getVoidTy();
  mlir::Value address = builder.createPtrBitcast(ops[0], voidTy);
  bool isWrite{};
  int locality{};

  assert(builtinID == X86::BI_mm_prefetch || builtinID == X86::BI_m_prefetchw ||
         builtinID == X86::BI_m_prefetch && "Expected prefetch builtin");

  if (builtinID == X86::BI_mm_prefetch) {
    int hint = cgf.getSExtIntValueFromConstOp(ops[1]);
    isWrite = (hint >> 2) & 0x1;
    locality = hint & 0x3;
  } else {
    isWrite = (builtinID == X86::BI_m_prefetchw);
    locality = 0x3;
  }

  cir::PrefetchOp::create(builder, location, address, locality, isWrite);
  return {};
}

static mlir::Value emitX86CompressExpand(CIRGenBuilderTy &builder,
                                         mlir::Location loc, mlir::Value source,
                                         mlir::Value mask,
                                         mlir::Value inputVector,
                                         const std::string &id) {
  auto resultTy = cast<cir::VectorType>(mask.getType());
  mlir::Value maskValue = getMaskVecValue(
      builder, loc, inputVector, cast<cir::VectorType>(resultTy).getSize());
  return builder.emitIntrinsicCallOp(loc, id, resultTy,
                                     mlir::ValueRange{source, mask, maskValue});
}

static mlir::Value emitX86Select(CIRGenBuilderTy &builder, mlir::Location loc,
                                 mlir::Value mask, mlir::Value op0,
                                 mlir::Value op1) {
  auto constOp = mlir::dyn_cast_or_null<cir::ConstantOp>(mask.getDefiningOp());
  // If the mask is all ones just return first argument.
  if (constOp && constOp.isAllOnesValue())
    return op0;

  mask = getMaskVecValue(builder, loc, mask,
                         cast<cir::VectorType>(op0.getType()).getSize());

  return cir::VecTernaryOp::create(builder, loc, mask, op0, op1);
}

static mlir::Value emitX86MaskAddLogic(CIRGenBuilderTy &builder,
                                       mlir::Location loc,
                                       const std::string &intrinsicName,
                                       SmallVectorImpl<mlir::Value> &ops) {

  auto intTy = cast<cir::IntType>(ops[0].getType());
  unsigned numElts = intTy.getWidth();
  mlir::Value lhsVec = getMaskVecValue(builder, loc, ops[0], numElts);
  mlir::Value rhsVec = getMaskVecValue(builder, loc, ops[1], numElts);
  mlir::Type vecTy = lhsVec.getType();
  mlir::Value resVec = builder.emitIntrinsicCallOp(
      loc, intrinsicName, vecTy, mlir::ValueRange{lhsVec, rhsVec});
  return builder.createBitcast(resVec, ops[0].getType());
}

static mlir::Value emitX86MaskUnpack(CIRGenBuilderTy &builder,
                                     mlir::Location loc,
                                     const std::string &intrinsicName,
                                     SmallVectorImpl<mlir::Value> &ops) {
  unsigned numElems = cast<cir::IntType>(ops[0].getType()).getWidth();

  // Convert both operands to mask vectors.
  mlir::Value lhs = getMaskVecValue(builder, loc, ops[0], numElems);
  mlir::Value rhs = getMaskVecValue(builder, loc, ops[1], numElems);

  mlir::Type i32Ty = builder.getSInt32Ty();

  // Create indices for extracting the first half of each vector.
  SmallVector<mlir::Attribute, 32> halfIndices;
  for (auto i : llvm::seq<unsigned>(0, numElems / 2))
    halfIndices.push_back(cir::IntAttr::get(i32Ty, i));

  // Extract first half of each vector. This gives better codegen than
  // doing it in a single shuffle.
  mlir::Value lhsHalf = builder.createVecShuffle(loc, lhs, lhs, halfIndices);
  mlir::Value rhsHalf = builder.createVecShuffle(loc, rhs, rhs, halfIndices);

  // Create indices for concatenating the vectors.
  // NOTE: Operands are swapped to match the intrinsic definition.
  // After the half extraction, both vectors have numElems/2 elements.
  // In createVecShuffle(rhsHalf, lhsHalf, indices), indices [0..numElems/2-1]
  // select from rhsHalf, and indices [numElems/2..numElems-1] select from
  // lhsHalf.
  SmallVector<mlir::Attribute, 64> concatIndices;
  for (auto i : llvm::seq<unsigned>(0, numElems))
    concatIndices.push_back(cir::IntAttr::get(i32Ty, i));

  // Concat the vectors (RHS first, then LHS).
  mlir::Value res =
      builder.createVecShuffle(loc, rhsHalf, lhsHalf, concatIndices);
  return builder.createBitcast(res, ops[0].getType());
}

static mlir::Value emitX86MaskLogic(CIRGenBuilderTy &builder,
                                    mlir::Location loc,
                                    cir::BinOpKind binOpKind,
                                    SmallVectorImpl<mlir::Value> &ops,
                                    bool invertLHS = false) {
  unsigned numElts = cast<cir::IntType>(ops[0].getType()).getWidth();
  mlir::Value lhs = getMaskVecValue(builder, loc, ops[0], numElts);
  mlir::Value rhs = getMaskVecValue(builder, loc, ops[1], numElts);

  if (invertLHS)
    lhs = builder.createNot(lhs);
  return builder.createBitcast(builder.createBinop(loc, lhs, binOpKind, rhs),
                               ops[0].getType());
}

static mlir::Value emitX86MaskTest(CIRGenBuilderTy &builder, mlir::Location loc,
                                   const std::string &intrinsicName,
                                   SmallVectorImpl<mlir::Value> &ops) {
  auto intTy = cast<cir::IntType>(ops[0].getType());
  unsigned numElts = intTy.getWidth();
  mlir::Value lhsVec = getMaskVecValue(builder, loc, ops[0], numElts);
  mlir::Value rhsVec = getMaskVecValue(builder, loc, ops[1], numElts);
  mlir::Type resTy = builder.getSInt32Ty();
  return builder.emitIntrinsicCallOp(loc, intrinsicName, resTy,
                                     mlir::ValueRange{lhsVec, rhsVec});
}

static mlir::Value emitX86MaskedCompareResult(CIRGenBuilderTy &builder,
                                              mlir::Value cmp, unsigned numElts,
                                              mlir::Value maskIn,
                                              mlir::Location loc) {
  if (maskIn) {
    auto c = mlir::dyn_cast_or_null<cir::ConstantOp>(maskIn.getDefiningOp());
    if (!c || !c.isAllOnesValue())
      cmp = builder.createAnd(loc, cmp,
                              getMaskVecValue(builder, loc, maskIn, numElts));
  }
  if (numElts < 8) {
    llvm::SmallVector<mlir::Attribute> indices;
    mlir::Type i64Ty = builder.getSInt64Ty();

    for (unsigned i = 0; i != numElts; ++i)
      indices.push_back(cir::IntAttr::get(i64Ty, i));
    for (unsigned i = numElts; i != 8; ++i)
      indices.push_back(cir::IntAttr::get(i64Ty, i % numElts + numElts));

    // This should shuffle between cmp (first vector) and null (second vector)
    mlir::Value nullVec = builder.getNullValue(cmp.getType(), loc);
    cmp = builder.createVecShuffle(loc, cmp, nullVec, indices);
  }
  return builder.createBitcast(cmp, builder.getUIntNTy(std::max(numElts, 8U)));
}

// TODO: The cgf parameter should be removed when all the NYI cases are
// implemented.
static std::optional<mlir::Value>
emitX86MaskedCompare(CIRGenFunction &cgf, CIRGenBuilderTy &builder, unsigned cc,
                     bool isSigned, ArrayRef<mlir::Value> ops,
                     mlir::Location loc) {
  assert((ops.size() == 2 || ops.size() == 4) &&
         "Unexpected number of arguments");
  unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
  mlir::Value cmp;

  if (cc == 3) {
    cgf.cgm.errorNYI(loc, "emitX86MaskedCompare: cc == 3");
    return {};
  } else if (cc == 7) {
    cgf.cgm.errorNYI(loc, "emitX86MaskedCompare cc == 7");
    return {};
  } else {
    cir::CmpOpKind pred;
    switch (cc) {
    default:
      llvm_unreachable("Unknown condition code");
    case 0:
      pred = cir::CmpOpKind::eq;
      break;
    case 1:
      pred = cir::CmpOpKind::lt;
      break;
    case 2:
      pred = cir::CmpOpKind::le;
      break;
    case 4:
      pred = cir::CmpOpKind::ne;
      break;
    case 5:
      pred = cir::CmpOpKind::ge;
      break;
    case 6:
      pred = cir::CmpOpKind::gt;
      break;
    }

    auto resultTy = cir::VectorType::get(builder.getSIntNTy(1), numElts);
    cmp = cir::VecCmpOp::create(builder, loc, resultTy, pred, ops[0], ops[1]);
  }

  mlir::Value maskIn;
  if (ops.size() == 4)
    maskIn = ops[3];

  return emitX86MaskedCompareResult(builder, cmp, numElts, maskIn, loc);
}

// TODO: The cgf parameter should be removed when all the NYI cases are
// implemented.
static std::optional<mlir::Value> emitX86ConvertToMask(CIRGenFunction &cgf,
                                                       CIRGenBuilderTy &builder,
                                                       mlir::Value in,
                                                       mlir::Location loc) {
  cir::ConstantOp zero = builder.getNullValue(in.getType(), loc);
  return emitX86MaskedCompare(cgf, builder, 1, true, {in, zero}, loc);
}

static std::optional<mlir::Value> emitX86SExtMask(CIRGenBuilderTy &builder,
                                                  mlir::Value op,
                                                  mlir::Type dstTy,
                                                  mlir::Location loc) {
  unsigned numberOfElements = cast<cir::VectorType>(dstTy).getSize();
  mlir::Value mask = getMaskVecValue(builder, loc, op, numberOfElements);

  return builder.createCast(loc, cir::CastKind::integral, mask, dstTy);
}

static mlir::Value emitVecInsert(CIRGenBuilderTy &builder, mlir::Location loc,
                                 mlir::Value vec, mlir::Value value,
                                 mlir::Value indexOp) {
  unsigned numElts = cast<cir::VectorType>(vec.getType()).getSize();

  uint64_t index =
      indexOp.getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

  index &= numElts - 1;

  cir::ConstantOp indexVal = builder.getUInt64(index, loc);

  return cir::VecInsertOp::create(builder, loc, vec, value, indexVal);
}

static mlir::Value emitX86FunnelShift(CIRGenBuilderTy &builder,
                                      mlir::Location location, mlir::Value &op0,
                                      mlir::Value &op1, mlir::Value &amt,
                                      bool isRight) {
  mlir::Type op0Ty = op0.getType();

  // Amount may be scalar immediate, in which case create a splat vector.
  // Funnel shifts amounts are treated as modulo and types are all power-of-2
  // so we only care about the lowest log2 bits anyway.
  if (amt.getType() != op0Ty) {
    auto vecTy = mlir::cast<cir::VectorType>(op0Ty);
    uint64_t numElems = vecTy.getSize();

    auto amtTy = mlir::cast<cir::IntType>(amt.getType());
    auto vecElemTy = mlir::cast<cir::IntType>(vecTy.getElementType());

    // If signed, cast to the same width but unsigned first to
    // ensure zero-extension when casting to a bigger unsigned `vecElemeTy`.
    if (amtTy.isSigned()) {
      cir::IntType unsignedAmtTy = builder.getUIntNTy(amtTy.getWidth());
      amt = builder.createIntCast(amt, unsignedAmtTy);
    }
    cir::IntType unsignedVecElemType = builder.getUIntNTy(vecElemTy.getWidth());
    amt = builder.createIntCast(amt, unsignedVecElemType);
    amt = cir::VecSplatOp::create(
        builder, location, cir::VectorType::get(unsignedVecElemType, numElems),
        amt);
  }

  const StringRef intrinsicName = isRight ? "fshr" : "fshl";
  return builder.emitIntrinsicCallOp(location, intrinsicName, op0Ty,
                                     mlir::ValueRange{op0, op1, amt});
}

static mlir::Value emitX86Muldq(CIRGenBuilderTy &builder, mlir::Location loc,
                                bool isSigned,
                                SmallVectorImpl<mlir::Value> &ops,
                                unsigned opTypePrimitiveSizeInBits) {
  mlir::Type ty = cir::VectorType::get(builder.getSInt64Ty(),
                                       opTypePrimitiveSizeInBits / 64);
  mlir::Value lhs = builder.createBitcast(loc, ops[0], ty);
  mlir::Value rhs = builder.createBitcast(loc, ops[1], ty);
  if (isSigned) {
    cir::ConstantOp shiftAmt =
        builder.getConstant(loc, cir::IntAttr::get(builder.getSInt64Ty(), 32));
    cir::VecSplatOp shiftSplatVecOp =
        cir::VecSplatOp::create(builder, loc, ty, shiftAmt.getResult());
    mlir::Value shiftSplatValue = shiftSplatVecOp.getResult();
    // In CIR, right-shift operations are automatically lowered to either an
    // arithmetic or logical shift depending on the operand type. The purpose
    // of the shifts here is to propagate the sign bit of the 32-bit input
    // into the upper bits of each vector lane.
    lhs = builder.createShift(loc, lhs, shiftSplatValue, true);
    lhs = builder.createShift(loc, lhs, shiftSplatValue, false);
    rhs = builder.createShift(loc, rhs, shiftSplatValue, true);
    rhs = builder.createShift(loc, rhs, shiftSplatValue, false);
  } else {
    cir::ConstantOp maskScalar = builder.getConstant(
        loc, cir::IntAttr::get(builder.getSInt64Ty(), 0xffffffff));
    cir::VecSplatOp mask =
        cir::VecSplatOp::create(builder, loc, ty, maskScalar.getResult());
    // Clear the upper bits
    lhs = builder.createAnd(loc, lhs, mask);
    rhs = builder.createAnd(loc, rhs, mask);
  }
  return builder.createMul(loc, lhs, rhs);
}

// Convert f16 half values to floats.
static mlir::Value emitX86CvtF16ToFloatExpr(CIRGenBuilderTy &builder,
                                            mlir::Location loc,
                                            llvm::ArrayRef<mlir::Value> ops,
                                            mlir::Type dstTy) {
  assert((ops.size() == 1 || ops.size() == 3 || ops.size() == 4) &&
         "Unknown cvtph2ps intrinsic");

  // If the SAE intrinsic doesn't use default rounding then we can't upgrade.
  if (ops.size() == 4) {
    auto constOp = ops[3].getDefiningOp<cir::ConstantOp>();
    assert(constOp && "Expected constant operand");
    if (constOp.getIntValue().getZExtValue() != 4) {
      return builder.emitIntrinsicCallOp(loc, "x86.avx512.mask.vcvtph2ps.512",
                                         dstTy, ops);
    }
  }

  unsigned numElts = cast<cir::VectorType>(dstTy).getSize();
  mlir::Value src = ops[0];

  // Extract the subvector
  if (numElts != cast<cir::VectorType>(src.getType()).getSize()) {
    assert(numElts == 4 && "Unexpected vector size");
    src = builder.createVecShuffle(loc, src, {0, 1, 2, 3});
  }

  // Bitcast from vXi16 to vXf16.
  cir::VectorType halfTy =
      cir::VectorType::get(cir::FP16Type::get(builder.getContext()), numElts);

  src = builder.createCast(cir::CastKind::bitcast, src, halfTy);

  // Perform the fp-extension
  mlir::Value res = builder.createCast(cir::CastKind::floating, src, dstTy);

  if (ops.size() >= 3)
    res = emitX86Select(builder, loc, ops[2], res, ops[1]);
  return res;
}

static mlir::Value emitX86vpcom(CIRGenBuilderTy &builder, mlir::Location loc,
                                llvm::SmallVector<mlir::Value> ops,
                                bool isSigned) {
  mlir::Value op0 = ops[0];
  mlir::Value op1 = ops[1];

  cir::VectorType ty = cast<cir::VectorType>(op0.getType());
  cir::IntType elementTy = cast<cir::IntType>(ty.getElementType());

  uint64_t imm = CIRGenFunction::getZExtIntValueFromConstOp(ops[2]) & 0x7;

  cir::CmpOpKind pred;
  switch (imm) {
  case 0x0:
    pred = cir::CmpOpKind::lt;
    break;
  case 0x1:
    pred = cir::CmpOpKind::le;
    break;
  case 0x2:
    pred = cir::CmpOpKind::gt;
    break;
  case 0x3:
    pred = cir::CmpOpKind::ge;
    break;
  case 0x4:
    pred = cir::CmpOpKind::eq;
    break;
  case 0x5:
    pred = cir::CmpOpKind::ne;
    break;
  case 0x6:
    return builder.getNullValue(ty, loc); // FALSE
  case 0x7: {
    llvm::APInt allOnes = llvm::APInt::getAllOnes(elementTy.getWidth());
    return cir::VecSplatOp::create(
        builder, loc, ty,
        builder.getConstAPInt(loc, elementTy, allOnes)); // TRUE
  }
  default:
    llvm_unreachable("Unexpected XOP vpcom/vpcomu predicate");
  }

  if ((!isSigned && elementTy.isSigned()) ||
      (isSigned && elementTy.isUnsigned())) {
    elementTy = elementTy.isSigned() ? builder.getUIntNTy(elementTy.getWidth())
                                     : builder.getSIntNTy(elementTy.getWidth());
    ty = cir::VectorType::get(elementTy, ty.getSize());
    op0 = builder.createBitcast(op0, ty);
    op1 = builder.createBitcast(op1, ty);
  }

  return builder.createVecCompare(loc, pred, op0, op1);
}

static mlir::Value emitX86Fpclass(CIRGenBuilderTy &builder, mlir::Location loc,
                                  unsigned builtinID,
                                  SmallVectorImpl<mlir::Value> &ops) {
  unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
  mlir::Value maskIn = ops[2];
  ops.erase(ops.begin() + 2);

  StringRef intrinsicName;
  switch (builtinID) {
  default:
    llvm_unreachable("Unsupported fpclass builtin");
  case X86::BI__builtin_ia32_vfpclassbf16128_mask:
    intrinsicName = "x86.avx10.fpclass.bf16.128";
    break;
  case X86::BI__builtin_ia32_vfpclassbf16256_mask:
    intrinsicName = "x86.avx10.fpclass.bf16.256";
    break;
  case X86::BI__builtin_ia32_vfpclassbf16512_mask:
    intrinsicName = "x86.avx10.fpclass.bf16.512";
    break;
  case X86::BI__builtin_ia32_fpclassph128_mask:
    intrinsicName = "x86.avx512fp16.fpclass.ph.128";
    break;
  case X86::BI__builtin_ia32_fpclassph256_mask:
    intrinsicName = "x86.avx512fp16.fpclass.ph.256";
    break;
  case X86::BI__builtin_ia32_fpclassph512_mask:
    intrinsicName = "x86.avx512fp16.fpclass.ph.512";
    break;
  case X86::BI__builtin_ia32_fpclassps128_mask:
    intrinsicName = "x86.avx512.fpclass.ps.128";
    break;
  case X86::BI__builtin_ia32_fpclassps256_mask:
    intrinsicName = "x86.avx512.fpclass.ps.256";
    break;
  case X86::BI__builtin_ia32_fpclassps512_mask:
    intrinsicName = "x86.avx512.fpclass.ps.512";
    break;
  case X86::BI__builtin_ia32_fpclasspd128_mask:
    intrinsicName = "x86.avx512.fpclass.pd.128";
    break;
  case X86::BI__builtin_ia32_fpclasspd256_mask:
    intrinsicName = "x86.avx512.fpclass.pd.256";
    break;
  case X86::BI__builtin_ia32_fpclasspd512_mask:
    intrinsicName = "x86.avx512.fpclass.pd.512";
    break;
  }

  auto cmpResultTy = cir::VectorType::get(builder.getSIntNTy(1), numElts);
  mlir::Value fpclass =
      builder.emitIntrinsicCallOp(loc, intrinsicName, cmpResultTy, ops);
  return emitX86MaskedCompareResult(builder, fpclass, numElts, maskIn, loc);
}

static mlir::Value emitX86Aes(CIRGenBuilderTy &builder, mlir::Location loc,
                              llvm::StringRef intrinsicName, mlir::Type retType,
                              llvm::ArrayRef<mlir::Value> ops) {
  // Create return struct type and call intrinsic function.
  mlir::Type vecType =
      mlir::cast<cir::PointerType>(ops[0].getType()).getPointee();
  cir::RecordType rstRecTy = builder.getAnonRecordTy({retType, vecType});
  mlir::Value rstValueRec = builder.emitIntrinsicCallOp(
      loc, intrinsicName, rstRecTy, mlir::ValueRange{ops[1], ops[2]});

  // Extract the first return value and truncate it to 1 bit, then cast result
  // to bool value.
  mlir::Value flag =
      cir::ExtractMemberOp::create(builder, loc, rstValueRec, /*index=*/0);
  mlir::Value flagBit0 = builder.createCast(loc, cir::CastKind::integral, flag,
                                            builder.getUIntNTy(1));
  mlir::Value succ = builder.createCast(loc, cir::CastKind::int_to_bool,
                                        flagBit0, builder.getBoolTy());

  // Extract the second return value, store it to output address if success.
  mlir::Value out =
      cir::ExtractMemberOp::create(builder, loc, rstValueRec, /*index=*/1);
  Address outAddr(ops[0], /*align=*/CharUnits::fromQuantity(16));
  cir::IfOp::create(
      builder, loc, succ, /*withElseRegion=*/true,
      /*thenBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location) {
        builder.createStore(loc, out, outAddr);
        builder.createYield(loc);
      },
      /*elseBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location) {
        mlir::Value zero = builder.getNullValue(vecType, loc);
        builder.createStore(loc, zero, outAddr);
        builder.createYield(loc);
      });

  return cir::ExtractMemberOp::create(builder, loc, rstValueRec, /*index=*/0);
}

static mlir::Value emitX86Aeswide(CIRGenBuilderTy &builder, mlir::Location loc,
                                  llvm::StringRef intrinsicName,
                                  mlir::Type retType,
                                  llvm::ArrayRef<mlir::Value> ops) {
  mlir::Type vecType =
      mlir::cast<cir::PointerType>(ops[1].getType()).getPointee();

  // Create struct for return type and load input arguments, then call
  // intrinsic function.
  mlir::Type recTypes[9] = {retType, vecType, vecType, vecType, vecType,
                            vecType, vecType, vecType, vecType};
  mlir::Value arguments[9];
  arguments[0] = ops[2];
  for (int i = 0; i < 8; i++) {
    // Loading each vector argument from input address.
    cir::ConstantOp idx = builder.getUInt32(i, loc);
    mlir::Value nextInElePtr =
        builder.getArrayElement(loc, loc, ops[1], vecType, idx,
                                /*shouldDecay=*/false);
    arguments[i + 1] =
        builder.createAlignedLoad(loc, vecType, nextInElePtr,
                                  /*align=*/CharUnits::fromQuantity(16));
  }
  cir::RecordType rstRecTy = builder.getAnonRecordTy(recTypes);
  mlir::Value rstValueRec =
      builder.emitIntrinsicCallOp(loc, intrinsicName, rstRecTy, arguments);

  // Extract the first return value and truncate it to 1 bit, then cast result
  // to bool value.
  mlir::Value flag =
      cir::ExtractMemberOp::create(builder, loc, rstValueRec, /*index=*/0);
  mlir::Value flagBit0 = builder.createCast(loc, cir::CastKind::integral, flag,
                                            builder.getUIntNTy(1));
  mlir::Value succ = builder.createCast(loc, cir::CastKind::int_to_bool,
                                        flagBit0, builder.getBoolTy());

  // Extract other return values, store those to output address if success.
  cir::IfOp::create(
      builder, loc, succ, /*withElseRegion=*/true,
      /*thenBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location) {
        for (int i = 0; i < 8; i++) {
          mlir::Value out =
              cir::ExtractMemberOp::create(builder, loc, rstValueRec,
                                           /*index=*/i + 1);
          cir::ConstantOp idx = builder.getUInt32(i, loc);
          mlir::Value nextOutEleAddr =
              builder.getArrayElement(loc, loc, ops[0], vecType, idx,
                                      /*shouldDecay=*/false);
          Address outAddr(nextOutEleAddr,
                          /*align=*/CharUnits::fromQuantity(16));
          builder.createStore(loc, out, outAddr);
        }
        builder.createYield(loc);
      },
      /*elseBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location) {
        mlir::Value zero = builder.getNullValue(vecType, loc);
        for (int i = 0; i < 8; i++) {
          cir::ConstantOp idx = builder.getUInt32(i, loc);
          mlir::Value nextOutEleAddr =
              builder.getArrayElement(loc, loc, ops[0], vecType, idx,
                                      /*shouldDecay=*/false);
          Address outAddr(nextOutEleAddr,
                          /*align=*/CharUnits::fromQuantity(16));
          builder.createStore(loc, zero, outAddr);
        }
        builder.createYield(loc);
      });

  return cir::ExtractMemberOp::create(builder, loc, rstValueRec, /*index=*/0);
}

std::optional<mlir::Value>
CIRGenFunction::emitX86BuiltinExpr(unsigned builtinID, const CallExpr *expr) {
  if (builtinID == Builtin::BI__builtin_cpu_is) {
    cgm.errorNYI(expr->getSourceRange(), "__builtin_cpu_is");
    return mlir::Value{};
  }
  if (builtinID == Builtin::BI__builtin_cpu_supports) {
    cgm.errorNYI(expr->getSourceRange(), "__builtin_cpu_supports");
    return mlir::Value{};
  }
  if (builtinID == Builtin::BI__builtin_cpu_init) {
    cgm.errorNYI(expr->getSourceRange(), "__builtin_cpu_init");
    return mlir::Value{};
  }

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  assert(!cir::MissingFeatures::msvcBuiltins());

  // Find out if any arguments are required to be integer constant expressions.
  assert(!cir::MissingFeatures::handleBuiltinICEArguments());

  // The operands of the builtin call
  llvm::SmallVector<mlir::Value> ops;

  // `ICEArguments` is a bitmap indicating whether the argument at the i-th bit
  // is required to be a constant integer expression.
  unsigned iceArguments = 0;
  ASTContext::GetBuiltinTypeError error;
  getContext().GetBuiltinType(builtinID, error, &iceArguments);
  assert(error == ASTContext::GE_None && "Error while getting builtin type.");

  for (auto [idx, arg] : llvm::enumerate(expr->arguments()))
    ops.push_back(emitScalarOrConstFoldImmArg(iceArguments, idx, arg));

  CIRGenBuilderTy &builder = getBuilder();
  mlir::Type voidTy = builder.getVoidTy();

  switch (builtinID) {
  default:
    return std::nullopt;
  case X86::BI_mm_clflush:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "x86.sse2.clflush", voidTy, ops[0]);
  case X86::BI_mm_lfence:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "x86.sse2.lfence", voidTy);
  case X86::BI_mm_pause:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "x86.sse2.pause", voidTy);
  case X86::BI_mm_mfence:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "x86.sse2.mfence", voidTy);
  case X86::BI_mm_sfence:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "x86.sse.sfence", voidTy);
  case X86::BI_mm_prefetch:
  case X86::BI_m_prefetch:
  case X86::BI_m_prefetchw:
    return emitPrefetch(*this, builtinID, expr, ops);
  case X86::BI__rdtsc:
  case X86::BI__builtin_ia32_rdtscp: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  case X86::BI__builtin_ia32_lzcnt_u16:
  case X86::BI__builtin_ia32_lzcnt_u32:
  case X86::BI__builtin_ia32_lzcnt_u64: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Value isZeroPoison = builder.getFalse(loc);
    return builder.emitIntrinsicCallOp(loc, "ctlz", ops[0].getType(),
                                       mlir::ValueRange{ops[0], isZeroPoison});
  }
  case X86::BI__builtin_ia32_tzcnt_u16:
  case X86::BI__builtin_ia32_tzcnt_u32:
  case X86::BI__builtin_ia32_tzcnt_u64: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Value isZeroPoison = builder.getFalse(loc);
    return builder.emitIntrinsicCallOp(loc, "cttz", ops[0].getType(),
                                       mlir::ValueRange{ops[0], isZeroPoison});
  }
  case X86::BI__builtin_ia32_undef128:
  case X86::BI__builtin_ia32_undef256:
  case X86::BI__builtin_ia32_undef512:
    // The x86 definition of "undef" is not the same as the LLVM definition
    // (PR32176). We leave optimizing away an unnecessary zero constant to the
    // IR optimizer and backend.
    // TODO: If we had a "freeze" IR instruction to generate a fixed undef
    //  value, we should use that here instead of a zero.
    return builder.getNullValue(convertType(expr->getType()),
                                getLoc(expr->getExprLoc()));
  case X86::BI__builtin_ia32_vec_ext_v4hi:
  case X86::BI__builtin_ia32_vec_ext_v16qi:
  case X86::BI__builtin_ia32_vec_ext_v8hi:
  case X86::BI__builtin_ia32_vec_ext_v4si:
  case X86::BI__builtin_ia32_vec_ext_v4sf:
  case X86::BI__builtin_ia32_vec_ext_v2di:
  case X86::BI__builtin_ia32_vec_ext_v32qi:
  case X86::BI__builtin_ia32_vec_ext_v16hi:
  case X86::BI__builtin_ia32_vec_ext_v8si:
  case X86::BI__builtin_ia32_vec_ext_v4di: {
    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();

    uint64_t index = getZExtIntValueFromConstOp(ops[1]);
    index &= numElts - 1;

    cir::ConstantOp indexVal =
        builder.getUInt64(index, getLoc(expr->getExprLoc()));

    // These builtins exist so we can ensure the index is an ICE and in range.
    // Otherwise we could just do this in the header file.
    return cir::VecExtractOp::create(builder, getLoc(expr->getExprLoc()),
                                     ops[0], indexVal);
  }
  case X86::BI__builtin_ia32_vec_set_v4hi:
  case X86::BI__builtin_ia32_vec_set_v16qi:
  case X86::BI__builtin_ia32_vec_set_v8hi:
  case X86::BI__builtin_ia32_vec_set_v4si:
  case X86::BI__builtin_ia32_vec_set_v2di:
  case X86::BI__builtin_ia32_vec_set_v32qi:
  case X86::BI__builtin_ia32_vec_set_v16hi:
  case X86::BI__builtin_ia32_vec_set_v8si:
  case X86::BI__builtin_ia32_vec_set_v4di: {
    return emitVecInsert(builder, getLoc(expr->getExprLoc()), ops[0], ops[1],
                         ops[2]);
  }
  case X86::BI__builtin_ia32_kunpckhi:
    return emitX86MaskUnpack(builder, getLoc(expr->getExprLoc()),
                             "x86.avx512.kunpackb", ops);
  case X86::BI__builtin_ia32_kunpcksi:
    return emitX86MaskUnpack(builder, getLoc(expr->getExprLoc()),
                             "x86.avx512.kunpackw", ops);
  case X86::BI__builtin_ia32_kunpckdi:
    return emitX86MaskUnpack(builder, getLoc(expr->getExprLoc()),
                             "x86.avx512.kunpackd", ops);
  case X86::BI_mm_setcsr:
  case X86::BI__builtin_ia32_ldmxcsr: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    Address tmp = createMemTemp(expr->getArg(0)->getType(), loc);
    builder.createStore(loc, ops[0], tmp);
    return builder.emitIntrinsicCallOp(loc, "x86.sse.ldmxcsr",
                                       builder.getVoidTy(), tmp.getPointer());
  }
  case X86::BI_mm_getcsr:
  case X86::BI__builtin_ia32_stmxcsr: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    Address tmp = createMemTemp(expr->getType(), loc);
    builder.emitIntrinsicCallOp(loc, "x86.sse.stmxcsr", builder.getVoidTy(),
                                tmp.getPointer());
    return builder.createLoad(loc, tmp);
  }
  case X86::BI__builtin_ia32_xsave:
  case X86::BI__builtin_ia32_xsave64:
  case X86::BI__builtin_ia32_xrstor:
  case X86::BI__builtin_ia32_xrstor64:
  case X86::BI__builtin_ia32_xsaveopt:
  case X86::BI__builtin_ia32_xsaveopt64:
  case X86::BI__builtin_ia32_xrstors:
  case X86::BI__builtin_ia32_xrstors64:
  case X86::BI__builtin_ia32_xsavec:
  case X86::BI__builtin_ia32_xsavec64:
  case X86::BI__builtin_ia32_xsaves:
  case X86::BI__builtin_ia32_xsaves64:
  case X86::BI__builtin_ia32_xsetbv:
  case X86::BI_xsetbv: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_xsave:
      intrinsicName = "x86.xsave";
      break;
    case X86::BI__builtin_ia32_xsave64:
      intrinsicName = "x86.xsave64";
      break;
    case X86::BI__builtin_ia32_xrstor:
      intrinsicName = "x86.xrstor";
      break;
    case X86::BI__builtin_ia32_xrstor64:
      intrinsicName = "x86.xrstor64";
      break;
    case X86::BI__builtin_ia32_xsaveopt:
      intrinsicName = "x86.xsaveopt";
      break;
    case X86::BI__builtin_ia32_xsaveopt64:
      intrinsicName = "x86.xsaveopt64";
      break;
    case X86::BI__builtin_ia32_xrstors:
      intrinsicName = "x86.xrstors";
      break;
    case X86::BI__builtin_ia32_xrstors64:
      intrinsicName = "x86.xrstors64";
      break;
    case X86::BI__builtin_ia32_xsavec:
      intrinsicName = "x86.xsavec";
      break;
    case X86::BI__builtin_ia32_xsavec64:
      intrinsicName = "x86.xsavec64";
      break;
    case X86::BI__builtin_ia32_xsaves:
      intrinsicName = "x86.xsaves";
      break;
    case X86::BI__builtin_ia32_xsaves64:
      intrinsicName = "x86.xsaves64";
      break;
    case X86::BI__builtin_ia32_xsetbv:
    case X86::BI_xsetbv:
      intrinsicName = "x86.xsetbv";
      break;
    }

    // The xsave family of instructions take a 64-bit mask that specifies
    // which processor state components to save/restore. The hardware expects
    // this mask split into two 32-bit registers: EDX (high 32 bits) and
    // EAX (low 32 bits).
    mlir::Type i32Ty = builder.getSInt32Ty();

    // Mhi = (uint32_t)(ops[1] >> 32) - extract high 32 bits via right shift
    cir::ConstantOp shift32 = builder.getSInt64(32, loc);
    mlir::Value mhi = builder.createShift(loc, ops[1], shift32.getResult(),
                                          /*isShiftLeft=*/false);
    mhi = builder.createIntCast(mhi, i32Ty);

    // Mlo = (uint32_t)ops[1] - extract low 32 bits by truncation
    mlir::Value mlo = builder.createIntCast(ops[1], i32Ty);

    return builder.emitIntrinsicCallOp(loc, intrinsicName, voidTy,
                                       mlir::ValueRange{ops[0], mhi, mlo});
  }
  case X86::BI__builtin_ia32_xgetbv:
  case X86::BI_xgetbv:
    // xgetbv reads the extended control register specified by ops[0] (ECX)
    // and returns the 64-bit value
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()), "x86.xgetbv",
                                       builder.getUInt64Ty(), ops[0]);
  case X86::BI__builtin_ia32_storedqudi128_mask:
  case X86::BI__builtin_ia32_storedqusi128_mask:
  case X86::BI__builtin_ia32_storedquhi128_mask:
  case X86::BI__builtin_ia32_storedquqi128_mask:
  case X86::BI__builtin_ia32_storeupd128_mask:
  case X86::BI__builtin_ia32_storeups128_mask:
  case X86::BI__builtin_ia32_storedqudi256_mask:
  case X86::BI__builtin_ia32_storedqusi256_mask:
  case X86::BI__builtin_ia32_storedquhi256_mask:
  case X86::BI__builtin_ia32_storedquqi256_mask:
  case X86::BI__builtin_ia32_storeupd256_mask:
  case X86::BI__builtin_ia32_storeups256_mask:
  case X86::BI__builtin_ia32_storedqudi512_mask:
  case X86::BI__builtin_ia32_storedqusi512_mask:
  case X86::BI__builtin_ia32_storedquhi512_mask:
  case X86::BI__builtin_ia32_storedquqi512_mask:
  case X86::BI__builtin_ia32_storeupd512_mask:
  case X86::BI__builtin_ia32_storeups512_mask:
  case X86::BI__builtin_ia32_storesbf16128_mask:
  case X86::BI__builtin_ia32_storesh128_mask:
  case X86::BI__builtin_ia32_storess128_mask:
  case X86::BI__builtin_ia32_storesd128_mask:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented x86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_cvtmask2b128:
  case X86::BI__builtin_ia32_cvtmask2b256:
  case X86::BI__builtin_ia32_cvtmask2b512:
  case X86::BI__builtin_ia32_cvtmask2w128:
  case X86::BI__builtin_ia32_cvtmask2w256:
  case X86::BI__builtin_ia32_cvtmask2w512:
  case X86::BI__builtin_ia32_cvtmask2d128:
  case X86::BI__builtin_ia32_cvtmask2d256:
  case X86::BI__builtin_ia32_cvtmask2d512:
  case X86::BI__builtin_ia32_cvtmask2q128:
  case X86::BI__builtin_ia32_cvtmask2q256:
  case X86::BI__builtin_ia32_cvtmask2q512:
    return emitX86SExtMask(this->getBuilder(), ops[0],
                           convertType(expr->getType()),
                           getLoc(expr->getExprLoc()));
  case X86::BI__builtin_ia32_cvtb2mask128:
  case X86::BI__builtin_ia32_cvtb2mask256:
  case X86::BI__builtin_ia32_cvtb2mask512:
  case X86::BI__builtin_ia32_cvtw2mask128:
  case X86::BI__builtin_ia32_cvtw2mask256:
  case X86::BI__builtin_ia32_cvtw2mask512:
  case X86::BI__builtin_ia32_cvtd2mask128:
  case X86::BI__builtin_ia32_cvtd2mask256:
  case X86::BI__builtin_ia32_cvtd2mask512:
  case X86::BI__builtin_ia32_cvtq2mask128:
  case X86::BI__builtin_ia32_cvtq2mask256:
  case X86::BI__builtin_ia32_cvtq2mask512:
    return emitX86ConvertToMask(*this, this->getBuilder(), ops[0],
                                getLoc(expr->getExprLoc()));
  case X86::BI__builtin_ia32_cvtdq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtdq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtqq2ph512_mask:
  case X86::BI__builtin_ia32_cvtudq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtuw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtudq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtuqq2ph512_mask:
  case X86::BI__builtin_ia32_vfmaddsh3_mask:
  case X86::BI__builtin_ia32_vfmaddss3_mask:
  case X86::BI__builtin_ia32_vfmaddsd3_mask:
  case X86::BI__builtin_ia32_vfmaddsh3_maskz:
  case X86::BI__builtin_ia32_vfmaddss3_maskz:
  case X86::BI__builtin_ia32_vfmaddsd3_maskz:
  case X86::BI__builtin_ia32_vfmaddsh3_mask3:
  case X86::BI__builtin_ia32_vfmaddss3_mask3:
  case X86::BI__builtin_ia32_vfmaddsd3_mask3:
  case X86::BI__builtin_ia32_vfmsubsh3_mask3:
  case X86::BI__builtin_ia32_vfmsubss3_mask3:
  case X86::BI__builtin_ia32_vfmsubsd3_mask3:
  case X86::BI__builtin_ia32_vfmaddph512_mask:
  case X86::BI__builtin_ia32_vfmaddph512_maskz:
  case X86::BI__builtin_ia32_vfmaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddps512_mask:
  case X86::BI__builtin_ia32_vfmaddps512_maskz:
  case X86::BI__builtin_ia32_vfmaddps512_mask3:
  case X86::BI__builtin_ia32_vfmsubps512_mask3:
  case X86::BI__builtin_ia32_vfmaddpd512_mask:
  case X86::BI__builtin_ia32_vfmaddpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubph512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubph512_mask:
  case X86::BI__builtin_ia32_vfmaddsubph512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubph512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask:
  case X86::BI__builtin_ia32_vfmaddsubps512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddps512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask:
  case X86::BI__builtin_ia32_vfmaddsubpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddpd512_mask3:
  case X86::BI__builtin_ia32_movdqa32store128_mask:
  case X86::BI__builtin_ia32_movdqa64store128_mask:
  case X86::BI__builtin_ia32_storeaps128_mask:
  case X86::BI__builtin_ia32_storeapd128_mask:
  case X86::BI__builtin_ia32_movdqa32store256_mask:
  case X86::BI__builtin_ia32_movdqa64store256_mask:
  case X86::BI__builtin_ia32_storeaps256_mask:
  case X86::BI__builtin_ia32_storeapd256_mask:
  case X86::BI__builtin_ia32_movdqa32store512_mask:
  case X86::BI__builtin_ia32_movdqa64store512_mask:
  case X86::BI__builtin_ia32_storeaps512_mask:
  case X86::BI__builtin_ia32_storeapd512_mask:
  case X86::BI__builtin_ia32_loadups128_mask:
  case X86::BI__builtin_ia32_loadups256_mask:
  case X86::BI__builtin_ia32_loadups512_mask:
  case X86::BI__builtin_ia32_loadupd128_mask:
  case X86::BI__builtin_ia32_loadupd256_mask:
  case X86::BI__builtin_ia32_loadupd512_mask:
  case X86::BI__builtin_ia32_loaddquqi128_mask:
  case X86::BI__builtin_ia32_loaddquqi256_mask:
  case X86::BI__builtin_ia32_loaddquqi512_mask:
  case X86::BI__builtin_ia32_loaddquhi128_mask:
  case X86::BI__builtin_ia32_loaddquhi256_mask:
  case X86::BI__builtin_ia32_loaddquhi512_mask:
  case X86::BI__builtin_ia32_loaddqusi128_mask:
  case X86::BI__builtin_ia32_loaddqusi256_mask:
  case X86::BI__builtin_ia32_loaddqusi512_mask:
  case X86::BI__builtin_ia32_loaddqudi128_mask:
  case X86::BI__builtin_ia32_loaddqudi256_mask:
  case X86::BI__builtin_ia32_loaddqudi512_mask:
  case X86::BI__builtin_ia32_loadsbf16128_mask:
  case X86::BI__builtin_ia32_loadsh128_mask:
  case X86::BI__builtin_ia32_loadss128_mask:
  case X86::BI__builtin_ia32_loadsd128_mask:
  case X86::BI__builtin_ia32_loadaps128_mask:
  case X86::BI__builtin_ia32_loadaps256_mask:
  case X86::BI__builtin_ia32_loadaps512_mask:
  case X86::BI__builtin_ia32_loadapd128_mask:
  case X86::BI__builtin_ia32_loadapd256_mask:
  case X86::BI__builtin_ia32_loadapd512_mask:
  case X86::BI__builtin_ia32_movdqa32load128_mask:
  case X86::BI__builtin_ia32_movdqa32load256_mask:
  case X86::BI__builtin_ia32_movdqa32load512_mask:
  case X86::BI__builtin_ia32_movdqa64load128_mask:
  case X86::BI__builtin_ia32_movdqa64load256_mask:
  case X86::BI__builtin_ia32_movdqa64load512_mask:
  case X86::BI__builtin_ia32_expandloaddf128_mask:
  case X86::BI__builtin_ia32_expandloaddf256_mask:
  case X86::BI__builtin_ia32_expandloaddf512_mask:
  case X86::BI__builtin_ia32_expandloadsf128_mask:
  case X86::BI__builtin_ia32_expandloadsf256_mask:
  case X86::BI__builtin_ia32_expandloadsf512_mask:
  case X86::BI__builtin_ia32_expandloaddi128_mask:
  case X86::BI__builtin_ia32_expandloaddi256_mask:
  case X86::BI__builtin_ia32_expandloaddi512_mask:
  case X86::BI__builtin_ia32_expandloadsi128_mask:
  case X86::BI__builtin_ia32_expandloadsi256_mask:
  case X86::BI__builtin_ia32_expandloadsi512_mask:
  case X86::BI__builtin_ia32_expandloadhi128_mask:
  case X86::BI__builtin_ia32_expandloadhi256_mask:
  case X86::BI__builtin_ia32_expandloadhi512_mask:
  case X86::BI__builtin_ia32_expandloadqi128_mask:
  case X86::BI__builtin_ia32_expandloadqi256_mask:
  case X86::BI__builtin_ia32_expandloadqi512_mask:
  case X86::BI__builtin_ia32_compressstoredf128_mask:
  case X86::BI__builtin_ia32_compressstoredf256_mask:
  case X86::BI__builtin_ia32_compressstoredf512_mask:
  case X86::BI__builtin_ia32_compressstoresf128_mask:
  case X86::BI__builtin_ia32_compressstoresf256_mask:
  case X86::BI__builtin_ia32_compressstoresf512_mask:
  case X86::BI__builtin_ia32_compressstoredi128_mask:
  case X86::BI__builtin_ia32_compressstoredi256_mask:
  case X86::BI__builtin_ia32_compressstoredi512_mask:
  case X86::BI__builtin_ia32_compressstoresi128_mask:
  case X86::BI__builtin_ia32_compressstoresi256_mask:
  case X86::BI__builtin_ia32_compressstoresi512_mask:
  case X86::BI__builtin_ia32_compressstorehi128_mask:
  case X86::BI__builtin_ia32_compressstorehi256_mask:
  case X86::BI__builtin_ia32_compressstorehi512_mask:
  case X86::BI__builtin_ia32_compressstoreqi128_mask:
  case X86::BI__builtin_ia32_compressstoreqi256_mask:
  case X86::BI__builtin_ia32_compressstoreqi512_mask:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_expanddf128_mask:
  case X86::BI__builtin_ia32_expanddf256_mask:
  case X86::BI__builtin_ia32_expanddf512_mask:
  case X86::BI__builtin_ia32_expandsf128_mask:
  case X86::BI__builtin_ia32_expandsf256_mask:
  case X86::BI__builtin_ia32_expandsf512_mask:
  case X86::BI__builtin_ia32_expanddi128_mask:
  case X86::BI__builtin_ia32_expanddi256_mask:
  case X86::BI__builtin_ia32_expanddi512_mask:
  case X86::BI__builtin_ia32_expandsi128_mask:
  case X86::BI__builtin_ia32_expandsi256_mask:
  case X86::BI__builtin_ia32_expandsi512_mask:
  case X86::BI__builtin_ia32_expandhi128_mask:
  case X86::BI__builtin_ia32_expandhi256_mask:
  case X86::BI__builtin_ia32_expandhi512_mask:
  case X86::BI__builtin_ia32_expandqi128_mask:
  case X86::BI__builtin_ia32_expandqi256_mask:
  case X86::BI__builtin_ia32_expandqi512_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    return emitX86CompressExpand(builder, loc, ops[0], ops[1], ops[2],
                                 "x86.avx512.mask.expand");
  }
  case X86::BI__builtin_ia32_compressdf128_mask:
  case X86::BI__builtin_ia32_compressdf256_mask:
  case X86::BI__builtin_ia32_compressdf512_mask:
  case X86::BI__builtin_ia32_compresssf128_mask:
  case X86::BI__builtin_ia32_compresssf256_mask:
  case X86::BI__builtin_ia32_compresssf512_mask:
  case X86::BI__builtin_ia32_compressdi128_mask:
  case X86::BI__builtin_ia32_compressdi256_mask:
  case X86::BI__builtin_ia32_compressdi512_mask:
  case X86::BI__builtin_ia32_compresssi128_mask:
  case X86::BI__builtin_ia32_compresssi256_mask:
  case X86::BI__builtin_ia32_compresssi512_mask:
  case X86::BI__builtin_ia32_compresshi128_mask:
  case X86::BI__builtin_ia32_compresshi256_mask:
  case X86::BI__builtin_ia32_compresshi512_mask:
  case X86::BI__builtin_ia32_compressqi128_mask:
  case X86::BI__builtin_ia32_compressqi256_mask:
  case X86::BI__builtin_ia32_compressqi512_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    return emitX86CompressExpand(builder, loc, ops[0], ops[1], ops[2],
                                 "x86.avx512.mask.compress");
  }
  case X86::BI__builtin_ia32_gather3div2df:
  case X86::BI__builtin_ia32_gather3div2di:
  case X86::BI__builtin_ia32_gather3div4df:
  case X86::BI__builtin_ia32_gather3div4di:
  case X86::BI__builtin_ia32_gather3div4sf:
  case X86::BI__builtin_ia32_gather3div4si:
  case X86::BI__builtin_ia32_gather3div8sf:
  case X86::BI__builtin_ia32_gather3div8si:
  case X86::BI__builtin_ia32_gather3siv2df:
  case X86::BI__builtin_ia32_gather3siv2di:
  case X86::BI__builtin_ia32_gather3siv4df:
  case X86::BI__builtin_ia32_gather3siv4di:
  case X86::BI__builtin_ia32_gather3siv4sf:
  case X86::BI__builtin_ia32_gather3siv4si:
  case X86::BI__builtin_ia32_gather3siv8sf:
  case X86::BI__builtin_ia32_gather3siv8si:
  case X86::BI__builtin_ia32_gathersiv8df:
  case X86::BI__builtin_ia32_gathersiv16sf:
  case X86::BI__builtin_ia32_gatherdiv8df:
  case X86::BI__builtin_ia32_gatherdiv16sf:
  case X86::BI__builtin_ia32_gathersiv8di:
  case X86::BI__builtin_ia32_gathersiv16si:
  case X86::BI__builtin_ia32_gatherdiv8di:
  case X86::BI__builtin_ia32_gatherdiv16si: {
    StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_gather3div2df:
      intrinsicName = "x86.avx512.mask.gather3div2.df";
      break;
    case X86::BI__builtin_ia32_gather3div2di:
      intrinsicName = "x86.avx512.mask.gather3div2.di";
      break;
    case X86::BI__builtin_ia32_gather3div4df:
      intrinsicName = "x86.avx512.mask.gather3div4.df";
      break;
    case X86::BI__builtin_ia32_gather3div4di:
      intrinsicName = "x86.avx512.mask.gather3div4.di";
      break;
    case X86::BI__builtin_ia32_gather3div4sf:
      intrinsicName = "x86.avx512.mask.gather3div4.sf";
      break;
    case X86::BI__builtin_ia32_gather3div4si:
      intrinsicName = "x86.avx512.mask.gather3div4.si";
      break;
    case X86::BI__builtin_ia32_gather3div8sf:
      intrinsicName = "x86.avx512.mask.gather3div8.sf";
      break;
    case X86::BI__builtin_ia32_gather3div8si:
      intrinsicName = "x86.avx512.mask.gather3div8.si";
      break;
    case X86::BI__builtin_ia32_gather3siv2df:
      intrinsicName = "x86.avx512.mask.gather3siv2.df";
      break;
    case X86::BI__builtin_ia32_gather3siv2di:
      intrinsicName = "x86.avx512.mask.gather3siv2.di";
      break;
    case X86::BI__builtin_ia32_gather3siv4df:
      intrinsicName = "x86.avx512.mask.gather3siv4.df";
      break;
    case X86::BI__builtin_ia32_gather3siv4di:
      intrinsicName = "x86.avx512.mask.gather3siv4.di";
      break;
    case X86::BI__builtin_ia32_gather3siv4sf:
      intrinsicName = "x86.avx512.mask.gather3siv4.sf";
      break;
    case X86::BI__builtin_ia32_gather3siv4si:
      intrinsicName = "x86.avx512.mask.gather3siv4.si";
      break;
    case X86::BI__builtin_ia32_gather3siv8sf:
      intrinsicName = "x86.avx512.mask.gather3siv8.sf";
      break;
    case X86::BI__builtin_ia32_gather3siv8si:
      intrinsicName = "x86.avx512.mask.gather3siv8.si";
      break;
    case X86::BI__builtin_ia32_gathersiv8df:
      intrinsicName = "x86.avx512.mask.gather.dpd.512";
      break;
    case X86::BI__builtin_ia32_gathersiv16sf:
      intrinsicName = "x86.avx512.mask.gather.dps.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv8df:
      intrinsicName = "x86.avx512.mask.gather.qpd.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv16sf:
      intrinsicName = "x86.avx512.mask.gather.qps.512";
      break;
    case X86::BI__builtin_ia32_gathersiv8di:
      intrinsicName = "x86.avx512.mask.gather.dpq.512";
      break;
    case X86::BI__builtin_ia32_gathersiv16si:
      intrinsicName = "x86.avx512.mask.gather.dpi.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv8di:
      intrinsicName = "x86.avx512.mask.gather.qpq.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv16si:
      intrinsicName = "x86.avx512.mask.gather.qpi.512";
      break;
    }

    mlir::Location loc = getLoc(expr->getExprLoc());
    unsigned minElts =
        std::min(cast<cir::VectorType>(ops[0].getType()).getSize(),
                 cast<cir::VectorType>(ops[2].getType()).getSize());
    ops[3] = getMaskVecValue(builder, loc, ops[3], minElts);
    return builder.emitIntrinsicCallOp(loc, intrinsicName,
                                       convertType(expr->getType()), ops);
  }
  case X86::BI__builtin_ia32_scattersiv8df:
  case X86::BI__builtin_ia32_scattersiv16sf:
  case X86::BI__builtin_ia32_scatterdiv8df:
  case X86::BI__builtin_ia32_scatterdiv16sf:
  case X86::BI__builtin_ia32_scattersiv8di:
  case X86::BI__builtin_ia32_scattersiv16si:
  case X86::BI__builtin_ia32_scatterdiv8di:
  case X86::BI__builtin_ia32_scatterdiv16si:
  case X86::BI__builtin_ia32_scatterdiv2df:
  case X86::BI__builtin_ia32_scatterdiv2di:
  case X86::BI__builtin_ia32_scatterdiv4df:
  case X86::BI__builtin_ia32_scatterdiv4di:
  case X86::BI__builtin_ia32_scatterdiv4sf:
  case X86::BI__builtin_ia32_scatterdiv4si:
  case X86::BI__builtin_ia32_scatterdiv8sf:
  case X86::BI__builtin_ia32_scatterdiv8si:
  case X86::BI__builtin_ia32_scattersiv2df:
  case X86::BI__builtin_ia32_scattersiv2di:
  case X86::BI__builtin_ia32_scattersiv4df:
  case X86::BI__builtin_ia32_scattersiv4di:
  case X86::BI__builtin_ia32_scattersiv4sf:
  case X86::BI__builtin_ia32_scattersiv4si:
  case X86::BI__builtin_ia32_scattersiv8sf:
  case X86::BI__builtin_ia32_scattersiv8si: {
    llvm::StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_scattersiv8df:
      intrinsicName = "x86.avx512.mask.scatter.dpd.512";
      break;
    case X86::BI__builtin_ia32_scattersiv16sf:
      intrinsicName = "x86.avx512.mask.scatter.dps.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv8df:
      intrinsicName = "x86.avx512.mask.scatter.qpd.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv16sf:
      intrinsicName = "x86.avx512.mask.scatter.qps.512";
      break;
    case X86::BI__builtin_ia32_scattersiv8di:
      intrinsicName = "x86.avx512.mask.scatter.dpq.512";
      break;
    case X86::BI__builtin_ia32_scattersiv16si:
      intrinsicName = "x86.avx512.mask.scatter.dpi.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv8di:
      intrinsicName = "x86.avx512.mask.scatter.qpq.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv16si:
      intrinsicName = "x86.avx512.mask.scatter.qpi.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv2df:
      intrinsicName = "x86.avx512.mask.scatterdiv2.df";
      break;
    case X86::BI__builtin_ia32_scatterdiv2di:
      intrinsicName = "x86.avx512.mask.scatterdiv2.di";
      break;
    case X86::BI__builtin_ia32_scatterdiv4df:
      intrinsicName = "x86.avx512.mask.scatterdiv4.df";
      break;
    case X86::BI__builtin_ia32_scatterdiv4di:
      intrinsicName = "x86.avx512.mask.scatterdiv4.di";
      break;
    case X86::BI__builtin_ia32_scatterdiv4sf:
      intrinsicName = "x86.avx512.mask.scatterdiv4.sf";
      break;
    case X86::BI__builtin_ia32_scatterdiv4si:
      intrinsicName = "x86.avx512.mask.scatterdiv4.si";
      break;
    case X86::BI__builtin_ia32_scatterdiv8sf:
      intrinsicName = "x86.avx512.mask.scatterdiv8.sf";
      break;
    case X86::BI__builtin_ia32_scatterdiv8si:
      intrinsicName = "x86.avx512.mask.scatterdiv8.si";
      break;
    case X86::BI__builtin_ia32_scattersiv2df:
      intrinsicName = "x86.avx512.mask.scattersiv2.df";
      break;
    case X86::BI__builtin_ia32_scattersiv2di:
      intrinsicName = "x86.avx512.mask.scattersiv2.di";
      break;
    case X86::BI__builtin_ia32_scattersiv4df:
      intrinsicName = "x86.avx512.mask.scattersiv4.df";
      break;
    case X86::BI__builtin_ia32_scattersiv4di:
      intrinsicName = "x86.avx512.mask.scattersiv4.di";
      break;
    case X86::BI__builtin_ia32_scattersiv4sf:
      intrinsicName = "x86.avx512.mask.scattersiv4.sf";
      break;
    case X86::BI__builtin_ia32_scattersiv4si:
      intrinsicName = "x86.avx512.mask.scattersiv4.si";
      break;
    case X86::BI__builtin_ia32_scattersiv8sf:
      intrinsicName = "x86.avx512.mask.scattersiv8.sf";
      break;
    case X86::BI__builtin_ia32_scattersiv8si:
      intrinsicName = "x86.avx512.mask.scattersiv8.si";
      break;
    }

    mlir::Location loc = getLoc(expr->getExprLoc());
    unsigned minElts =
        std::min(cast<cir::VectorType>(ops[2].getType()).getSize(),
                 cast<cir::VectorType>(ops[3].getType()).getSize());
    ops[1] = getMaskVecValue(builder, loc, ops[1], minElts);

    return builder.emitIntrinsicCallOp(loc, intrinsicName,
                                       convertType(expr->getType()), ops);
  }
  case X86::BI__builtin_ia32_vextractf128_pd256:
  case X86::BI__builtin_ia32_vextractf128_ps256:
  case X86::BI__builtin_ia32_vextractf128_si256:
  case X86::BI__builtin_ia32_extract128i256:
  case X86::BI__builtin_ia32_extractf64x4_mask:
  case X86::BI__builtin_ia32_extractf32x4_mask:
  case X86::BI__builtin_ia32_extracti64x4_mask:
  case X86::BI__builtin_ia32_extracti32x4_mask:
  case X86::BI__builtin_ia32_extractf32x8_mask:
  case X86::BI__builtin_ia32_extracti32x8_mask:
  case X86::BI__builtin_ia32_extractf32x4_256_mask:
  case X86::BI__builtin_ia32_extracti32x4_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_256_mask:
  case X86::BI__builtin_ia32_extracti64x2_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_512_mask:
  case X86::BI__builtin_ia32_extracti64x2_512_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    cir::VectorType dstTy = cast<cir::VectorType>(convertType(expr->getType()));
    unsigned numElts = dstTy.getSize();
    unsigned srcNumElts = cast<cir::VectorType>(ops[0].getType()).getSize();
    unsigned subVectors = srcNumElts / numElts;
    assert(llvm::isPowerOf2_32(subVectors) && "Expected power of 2 subvectors");
    unsigned index =
        ops[1].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

    index &= subVectors - 1; // Remove any extra bits.
    index *= numElts;

    int64_t indices[16];
    std::iota(indices, indices + numElts, index);

    mlir::Value poison =
        builder.getConstant(loc, cir::PoisonAttr::get(ops[0].getType()));
    mlir::Value res = builder.createVecShuffle(loc, ops[0], poison,
                                               ArrayRef(indices, numElts));
    if (ops.size() == 4)
      res = emitX86Select(builder, loc, ops[3], res, ops[2]);

    return res;
  }
  case X86::BI__builtin_ia32_vinsertf128_pd256:
  case X86::BI__builtin_ia32_vinsertf128_ps256:
  case X86::BI__builtin_ia32_vinsertf128_si256:
  case X86::BI__builtin_ia32_insert128i256:
  case X86::BI__builtin_ia32_insertf64x4:
  case X86::BI__builtin_ia32_insertf32x4:
  case X86::BI__builtin_ia32_inserti64x4:
  case X86::BI__builtin_ia32_inserti32x4:
  case X86::BI__builtin_ia32_insertf32x8:
  case X86::BI__builtin_ia32_inserti32x8:
  case X86::BI__builtin_ia32_insertf32x4_256:
  case X86::BI__builtin_ia32_inserti32x4_256:
  case X86::BI__builtin_ia32_insertf64x2_256:
  case X86::BI__builtin_ia32_inserti64x2_256:
  case X86::BI__builtin_ia32_insertf64x2_512:
  case X86::BI__builtin_ia32_inserti64x2_512: {
    unsigned dstNumElts = cast<cir::VectorType>(ops[0].getType()).getSize();
    unsigned srcNumElts = cast<cir::VectorType>(ops[1].getType()).getSize();
    unsigned subVectors = dstNumElts / srcNumElts;
    assert(llvm::isPowerOf2_32(subVectors) && "Expected power of 2 subvectors");
    assert(dstNumElts <= 16);

    uint64_t index = getZExtIntValueFromConstOp(ops[2]);
    index &= subVectors - 1; // Remove any extra bits.
    index *= srcNumElts;

    llvm::SmallVector<int64_t, 16> mask(dstNumElts);
    for (unsigned i = 0; i != dstNumElts; ++i)
      mask[i] = (i >= srcNumElts) ? srcNumElts + (i % srcNumElts) : i;

    mlir::Value op1 =
        builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[1], mask);

    for (unsigned i = 0; i != dstNumElts; ++i) {
      if (i >= index && i < (index + srcNumElts))
        mask[i] = (i - index) + dstNumElts;
      else
        mask[i] = i;
    }

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0], op1,
                                    mask);
  }
  case X86::BI__builtin_ia32_pmovqd512_mask:
  case X86::BI__builtin_ia32_pmovwb512_mask: {
    mlir::Value Res =
        builder.createIntCast(ops[0], cast<cir::VectorType>(ops[1].getType()));
    return emitX86Select(builder, getLoc(expr->getExprLoc()), ops[2], Res,
                         ops[1]);
  }
  case X86::BI__builtin_ia32_pblendw128:
  case X86::BI__builtin_ia32_blendpd:
  case X86::BI__builtin_ia32_blendps:
  case X86::BI__builtin_ia32_blendpd256:
  case X86::BI__builtin_ia32_blendps256:
  case X86::BI__builtin_ia32_pblendw256:
  case X86::BI__builtin_ia32_pblendd128:
  case X86::BI__builtin_ia32_pblendd256: {
    uint32_t imm = getZExtIntValueFromConstOp(ops[2]);
    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();

    llvm::SmallVector<mlir::Attribute, 16> indices;
    // If there are more than 8 elements, the immediate is used twice so make
    // sure we handle that.
    mlir::Type i32Ty = builder.getSInt32Ty();
    for (unsigned i = 0; i != numElts; ++i)
      indices.push_back(
          cir::IntAttr::get(i32Ty, ((imm >> (i % 8)) & 0x1) ? numElts + i : i));

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0], ops[1],
                                    indices);
  }
  case X86::BI__builtin_ia32_pshuflw:
  case X86::BI__builtin_ia32_pshuflw256:
  case X86::BI__builtin_ia32_pshuflw512:
    return emitPshufWord(builder, ops[0], ops[1], getLoc(expr->getExprLoc()),
                         true);
  case X86::BI__builtin_ia32_pshufhw:
  case X86::BI__builtin_ia32_pshufhw256:
  case X86::BI__builtin_ia32_pshufhw512:
    return emitPshufWord(builder, ops[0], ops[1], getLoc(expr->getExprLoc()),
                         false);
  case X86::BI__builtin_ia32_pshufd:
  case X86::BI__builtin_ia32_pshufd256:
  case X86::BI__builtin_ia32_pshufd512:
  case X86::BI__builtin_ia32_vpermilpd:
  case X86::BI__builtin_ia32_vpermilps:
  case X86::BI__builtin_ia32_vpermilpd256:
  case X86::BI__builtin_ia32_vpermilps256:
  case X86::BI__builtin_ia32_vpermilpd512:
  case X86::BI__builtin_ia32_vpermilps512: {
    const uint32_t imm = getSExtIntValueFromConstOp(ops[1]);

    llvm::SmallVector<int64_t, 16> mask(16);
    computeFullLaneShuffleMask(*this, ops[0], imm, false, mask);

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0], mask);
  }
  case X86::BI__builtin_ia32_shufpd:
  case X86::BI__builtin_ia32_shufpd256:
  case X86::BI__builtin_ia32_shufpd512:
  case X86::BI__builtin_ia32_shufps:
  case X86::BI__builtin_ia32_shufps256:
  case X86::BI__builtin_ia32_shufps512: {
    const uint32_t imm = getZExtIntValueFromConstOp(ops[2]);

    llvm::SmallVector<int64_t, 16> mask(16);
    computeFullLaneShuffleMask(*this, ops[0], imm, true, mask);

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0], ops[1],
                                    mask);
  }
  case X86::BI__builtin_ia32_permdi256:
  case X86::BI__builtin_ia32_permdf256:
  case X86::BI__builtin_ia32_permdi512:
  case X86::BI__builtin_ia32_permdf512: {
    unsigned imm =
        ops[1].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();
    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();

    // These intrinsics operate on 256-bit lanes of four 64-bit elements.
    int64_t Indices[8];

    for (unsigned l = 0; l != numElts; l += 4)
      for (unsigned i = 0; i != 4; ++i)
        Indices[l + i] = l + ((imm >> (2 * i)) & 0x3);

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0],
                                    ArrayRef(Indices, numElts));
  }
  case X86::BI__builtin_ia32_palignr128:
  case X86::BI__builtin_ia32_palignr256:
  case X86::BI__builtin_ia32_palignr512: {
    uint32_t shiftVal = getZExtIntValueFromConstOp(ops[2]) & 0xff;

    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
    assert(numElts % 16 == 0);

    // If palignr is shifting the pair of vectors more than the size of two
    // lanes, emit zero.
    if (shiftVal >= 32)
      return builder.getNullValue(convertType(expr->getType()),
                                  getLoc(expr->getExprLoc()));

    // If palignr is shifting the pair of input vectors more than one lane,
    // but less than two lanes, convert to shifting in zeroes.
    if (shiftVal > 16) {
      shiftVal -= 16;
      ops[1] = ops[0];
      ops[0] =
          builder.getNullValue(ops[0].getType(), getLoc(expr->getExprLoc()));
    }

    int64_t indices[64];
    // 256-bit palignr operates on 128-bit lanes so we need to handle that
    for (unsigned l = 0; l != numElts; l += 16) {
      for (unsigned i = 0; i != 16; ++i) {
        uint32_t idx = shiftVal + i;
        if (idx >= 16)
          idx += numElts - 16; // End of lane, switch operand.
        indices[l + i] = l + idx;
      }
    }

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[1], ops[0],
                                    ArrayRef(indices, numElts));
  }
  case X86::BI__builtin_ia32_alignd128:
  case X86::BI__builtin_ia32_alignd256:
  case X86::BI__builtin_ia32_alignd512:
  case X86::BI__builtin_ia32_alignq128:
  case X86::BI__builtin_ia32_alignq256:
  case X86::BI__builtin_ia32_alignq512: {
    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
    unsigned shiftVal =
        ops[2].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue() &
        0xff;

    // Mask the shift amount to width of a vector.
    shiftVal &= numElts - 1;

    SmallVector<mlir::Attribute, 16> indices;
    mlir::Type i32Ty = builder.getSInt32Ty();
    for (unsigned i = 0; i != numElts; ++i)
      indices.push_back(cir::IntAttr::get(i32Ty, i + shiftVal));

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), ops[0], ops[1],
                                    indices);
  }
  case X86::BI__builtin_ia32_shuf_f32x4_256:
  case X86::BI__builtin_ia32_shuf_f64x2_256:
  case X86::BI__builtin_ia32_shuf_i32x4_256:
  case X86::BI__builtin_ia32_shuf_i64x2_256:
  case X86::BI__builtin_ia32_shuf_f32x4:
  case X86::BI__builtin_ia32_shuf_f64x2:
  case X86::BI__builtin_ia32_shuf_i32x4:
  case X86::BI__builtin_ia32_shuf_i64x2: {
    mlir::Value src1 = ops[0];
    mlir::Value src2 = ops[1];

    unsigned imm =
        ops[2].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

    unsigned numElems = cast<cir::VectorType>(src1.getType()).getSize();
    unsigned totalBits = getContext().getTypeSize(expr->getArg(0)->getType());
    unsigned numLanes = totalBits == 512 ? 4 : 2;
    unsigned numElemsPerLane = numElems / numLanes;

    SmallVector<mlir::Attribute, 16> indices;
    mlir::Type i32Ty = builder.getSInt32Ty();

    for (unsigned l = 0; l != numElems; l += numElemsPerLane) {
      unsigned index = (imm % numLanes) * numElemsPerLane;
      imm /= numLanes;
      if (l >= (numElems / 2))
        index += numElems;
      for (unsigned i = 0; i != numElemsPerLane; ++i) {
        indices.push_back(cir::IntAttr::get(i32Ty, index + i));
      }
    }

    return builder.createVecShuffle(getLoc(expr->getExprLoc()), src1, src2,
                                    indices);
  }
  case X86::BI__builtin_ia32_vperm2f128_pd256:
  case X86::BI__builtin_ia32_vperm2f128_ps256:
  case X86::BI__builtin_ia32_vperm2f128_si256:
  case X86::BI__builtin_ia32_permti256:
  case X86::BI__builtin_ia32_pslldqi128_byteshift:
  case X86::BI__builtin_ia32_pslldqi256_byteshift:
  case X86::BI__builtin_ia32_pslldqi512_byteshift:
  case X86::BI__builtin_ia32_psrldqi128_byteshift:
  case X86::BI__builtin_ia32_psrldqi256_byteshift:
  case X86::BI__builtin_ia32_psrldqi512_byteshift:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_kshiftliqi:
  case X86::BI__builtin_ia32_kshiftlihi:
  case X86::BI__builtin_ia32_kshiftlisi:
  case X86::BI__builtin_ia32_kshiftlidi: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    unsigned shiftVal =
        ops[1].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue() &
        0xff;
    unsigned numElems = cast<cir::IntType>(ops[0].getType()).getWidth();

    if (shiftVal >= numElems)
      return builder.getNullValue(ops[0].getType(), loc);

    mlir::Value in = getMaskVecValue(builder, loc, ops[0], numElems);

    SmallVector<mlir::Attribute, 64> indices;
    mlir::Type i32Ty = builder.getSInt32Ty();
    for (auto i : llvm::seq<unsigned>(0, numElems))
      indices.push_back(cir::IntAttr::get(i32Ty, numElems + i - shiftVal));

    mlir::Value zero = builder.getNullValue(in.getType(), loc);
    mlir::Value sv = builder.createVecShuffle(loc, zero, in, indices);
    return builder.createBitcast(sv, ops[0].getType());
  }
  case X86::BI__builtin_ia32_kshiftriqi:
  case X86::BI__builtin_ia32_kshiftrihi:
  case X86::BI__builtin_ia32_kshiftrisi:
  case X86::BI__builtin_ia32_kshiftridi: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    unsigned shiftVal =
        ops[1].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue() &
        0xff;
    unsigned numElems = cast<cir::IntType>(ops[0].getType()).getWidth();

    if (shiftVal >= numElems)
      return builder.getNullValue(ops[0].getType(), loc);

    mlir::Value in = getMaskVecValue(builder, loc, ops[0], numElems);

    SmallVector<mlir::Attribute, 64> indices;
    mlir::Type i32Ty = builder.getSInt32Ty();
    for (auto i : llvm::seq<unsigned>(0, numElems))
      indices.push_back(cir::IntAttr::get(i32Ty, i + shiftVal));

    mlir::Value zero = builder.getNullValue(in.getType(), loc);
    mlir::Value sv = builder.createVecShuffle(loc, in, zero, indices);
    return builder.createBitcast(sv, ops[0].getType());
  }
  case X86::BI__builtin_ia32_vprotbi:
  case X86::BI__builtin_ia32_vprotwi:
  case X86::BI__builtin_ia32_vprotdi:
  case X86::BI__builtin_ia32_vprotqi:
  case X86::BI__builtin_ia32_prold128:
  case X86::BI__builtin_ia32_prold256:
  case X86::BI__builtin_ia32_prold512:
  case X86::BI__builtin_ia32_prolq128:
  case X86::BI__builtin_ia32_prolq256:
  case X86::BI__builtin_ia32_prolq512:
    return emitX86FunnelShift(builder, getLoc(expr->getExprLoc()), ops[0],
                              ops[0], ops[1], false);
  case X86::BI__builtin_ia32_prord128:
  case X86::BI__builtin_ia32_prord256:
  case X86::BI__builtin_ia32_prord512:
  case X86::BI__builtin_ia32_prorq128:
  case X86::BI__builtin_ia32_prorq256:
  case X86::BI__builtin_ia32_prorq512:
    return emitX86FunnelShift(builder, getLoc(expr->getExprLoc()), ops[0],
                              ops[0], ops[1], true);
  case X86::BI__builtin_ia32_selectb_128:
  case X86::BI__builtin_ia32_selectb_256:
  case X86::BI__builtin_ia32_selectb_512:
  case X86::BI__builtin_ia32_selectw_128:
  case X86::BI__builtin_ia32_selectw_256:
  case X86::BI__builtin_ia32_selectw_512:
  case X86::BI__builtin_ia32_selectd_128:
  case X86::BI__builtin_ia32_selectd_256:
  case X86::BI__builtin_ia32_selectd_512:
  case X86::BI__builtin_ia32_selectq_128:
  case X86::BI__builtin_ia32_selectq_256:
  case X86::BI__builtin_ia32_selectq_512:
  case X86::BI__builtin_ia32_selectph_128:
  case X86::BI__builtin_ia32_selectph_256:
  case X86::BI__builtin_ia32_selectph_512:
  case X86::BI__builtin_ia32_selectpbf_128:
  case X86::BI__builtin_ia32_selectpbf_256:
  case X86::BI__builtin_ia32_selectpbf_512:
  case X86::BI__builtin_ia32_selectps_128:
  case X86::BI__builtin_ia32_selectps_256:
  case X86::BI__builtin_ia32_selectps_512:
  case X86::BI__builtin_ia32_selectpd_128:
  case X86::BI__builtin_ia32_selectpd_256:
  case X86::BI__builtin_ia32_selectpd_512:
  case X86::BI__builtin_ia32_selectsh_128:
  case X86::BI__builtin_ia32_selectsbf_128:
  case X86::BI__builtin_ia32_selectss_128:
  case X86::BI__builtin_ia32_selectsd_128:
  case X86::BI__builtin_ia32_cmpb128_mask:
  case X86::BI__builtin_ia32_cmpb256_mask:
  case X86::BI__builtin_ia32_cmpb512_mask:
  case X86::BI__builtin_ia32_cmpw128_mask:
  case X86::BI__builtin_ia32_cmpw256_mask:
  case X86::BI__builtin_ia32_cmpw512_mask:
  case X86::BI__builtin_ia32_cmpd128_mask:
  case X86::BI__builtin_ia32_cmpd256_mask:
  case X86::BI__builtin_ia32_cmpd512_mask:
  case X86::BI__builtin_ia32_cmpq128_mask:
  case X86::BI__builtin_ia32_cmpq256_mask:
  case X86::BI__builtin_ia32_cmpq512_mask:
  case X86::BI__builtin_ia32_ucmpb128_mask:
  case X86::BI__builtin_ia32_ucmpb256_mask:
  case X86::BI__builtin_ia32_ucmpb512_mask:
  case X86::BI__builtin_ia32_ucmpw128_mask:
  case X86::BI__builtin_ia32_ucmpw256_mask:
  case X86::BI__builtin_ia32_ucmpw512_mask:
  case X86::BI__builtin_ia32_ucmpd128_mask:
  case X86::BI__builtin_ia32_ucmpd256_mask:
  case X86::BI__builtin_ia32_ucmpd512_mask:
  case X86::BI__builtin_ia32_ucmpq128_mask:
  case X86::BI__builtin_ia32_ucmpq256_mask:
  case X86::BI__builtin_ia32_ucmpq512_mask:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_vpcomb:
  case X86::BI__builtin_ia32_vpcomw:
  case X86::BI__builtin_ia32_vpcomd:
  case X86::BI__builtin_ia32_vpcomq:
    return emitX86vpcom(builder, getLoc(expr->getExprLoc()), ops, true);
  case X86::BI__builtin_ia32_vpcomub:
  case X86::BI__builtin_ia32_vpcomuw:
  case X86::BI__builtin_ia32_vpcomud:
  case X86::BI__builtin_ia32_vpcomuq:
    return emitX86vpcom(builder, getLoc(expr->getExprLoc()), ops, false);
  case X86::BI__builtin_ia32_kortestcqi:
  case X86::BI__builtin_ia32_kortestchi:
  case X86::BI__builtin_ia32_kortestcsi:
  case X86::BI__builtin_ia32_kortestcdi: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    cir::IntType ty = cast<cir::IntType>(ops[0].getType());
    mlir::Value allOnesOp =
        builder.getConstAPInt(loc, ty, APInt::getAllOnes(ty.getWidth()));
    mlir::Value orOp = emitX86MaskLogic(builder, loc, cir::BinOpKind::Or, ops);
    mlir::Value cmp =
        cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, orOp, allOnesOp);
    return builder.createCast(cir::CastKind::bool_to_int, cmp,
                              cgm.convertType(expr->getType()));
  }
  case X86::BI__builtin_ia32_kortestzqi:
  case X86::BI__builtin_ia32_kortestzhi:
  case X86::BI__builtin_ia32_kortestzsi:
  case X86::BI__builtin_ia32_kortestzdi: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    cir::IntType ty = cast<cir::IntType>(ops[0].getType());
    mlir::Value allZerosOp = builder.getNullValue(ty, loc).getResult();
    mlir::Value orOp = emitX86MaskLogic(builder, loc, cir::BinOpKind::Or, ops);
    mlir::Value cmp =
        cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, orOp, allZerosOp);
    return builder.createCast(cir::CastKind::bool_to_int, cmp,
                              cgm.convertType(expr->getType()));
  }
  case X86::BI__builtin_ia32_ktestcqi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestc.b", ops);
  case X86::BI__builtin_ia32_ktestzqi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestz.b", ops);
  case X86::BI__builtin_ia32_ktestchi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestc.w", ops);
  case X86::BI__builtin_ia32_ktestzhi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestz.w", ops);
  case X86::BI__builtin_ia32_ktestcsi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestc.d", ops);
  case X86::BI__builtin_ia32_ktestzsi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestz.d", ops);
  case X86::BI__builtin_ia32_ktestcdi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestc.q", ops);
  case X86::BI__builtin_ia32_ktestzdi:
    return emitX86MaskTest(builder, getLoc(expr->getExprLoc()),
                           "x86.avx512.ktestz.q", ops);
  case X86::BI__builtin_ia32_kaddqi:
    return emitX86MaskAddLogic(builder, getLoc(expr->getExprLoc()),
                               "x86.avx512.kadd.b", ops);
  case X86::BI__builtin_ia32_kaddhi:
    return emitX86MaskAddLogic(builder, getLoc(expr->getExprLoc()),
                               "x86.avx512.kadd.w", ops);
  case X86::BI__builtin_ia32_kaddsi:
    return emitX86MaskAddLogic(builder, getLoc(expr->getExprLoc()),
                               "x86.avx512.kadd.d", ops);
  case X86::BI__builtin_ia32_kadddi:
    return emitX86MaskAddLogic(builder, getLoc(expr->getExprLoc()),
                               "x86.avx512.kadd.q", ops);
  case X86::BI__builtin_ia32_kandqi:
  case X86::BI__builtin_ia32_kandhi:
  case X86::BI__builtin_ia32_kandsi:
  case X86::BI__builtin_ia32_kanddi:
    return emitX86MaskLogic(builder, getLoc(expr->getExprLoc()),
                            cir::BinOpKind::And, ops);
  case X86::BI__builtin_ia32_kandnqi:
  case X86::BI__builtin_ia32_kandnhi:
  case X86::BI__builtin_ia32_kandnsi:
  case X86::BI__builtin_ia32_kandndi:
    return emitX86MaskLogic(builder, getLoc(expr->getExprLoc()),
                            cir::BinOpKind::And, ops, true);
  case X86::BI__builtin_ia32_korqi:
  case X86::BI__builtin_ia32_korhi:
  case X86::BI__builtin_ia32_korsi:
  case X86::BI__builtin_ia32_kordi:
    return emitX86MaskLogic(builder, getLoc(expr->getExprLoc()),
                            cir::BinOpKind::Or, ops);
  case X86::BI__builtin_ia32_kxnorqi:
  case X86::BI__builtin_ia32_kxnorhi:
  case X86::BI__builtin_ia32_kxnorsi:
  case X86::BI__builtin_ia32_kxnordi:
    return emitX86MaskLogic(builder, getLoc(expr->getExprLoc()),
                            cir::BinOpKind::Xor, ops, true);
  case X86::BI__builtin_ia32_kxorqi:
  case X86::BI__builtin_ia32_kxorhi:
  case X86::BI__builtin_ia32_kxorsi:
  case X86::BI__builtin_ia32_kxordi:
    return emitX86MaskLogic(builder, getLoc(expr->getExprLoc()),
                            cir::BinOpKind::Xor, ops);
  case X86::BI__builtin_ia32_knotqi:
  case X86::BI__builtin_ia32_knothi:
  case X86::BI__builtin_ia32_knotsi:
  case X86::BI__builtin_ia32_knotdi: {
    cir::IntType intTy = cast<cir::IntType>(ops[0].getType());
    unsigned numElts = intTy.getWidth();
    mlir::Value resVec =
        getMaskVecValue(builder, getLoc(expr->getExprLoc()), ops[0], numElts);
    return builder.createBitcast(builder.createNot(resVec), ops[0].getType());
  }
  case X86::BI__builtin_ia32_kmovb:
  case X86::BI__builtin_ia32_kmovw:
  case X86::BI__builtin_ia32_kmovd:
  case X86::BI__builtin_ia32_kmovq: {
    // Bitcast to vXi1 type and then back to integer. This gets the mask
    // register type into the IR, but might be optimized out depending on
    // what's around it.
    cir::IntType intTy = cast<cir::IntType>(ops[0].getType());
    unsigned numElts = intTy.getWidth();
    mlir::Value resVec =
        getMaskVecValue(builder, getLoc(expr->getExprLoc()), ops[0], numElts);
    return builder.createBitcast(resVec, ops[0].getType());
  }
  case X86::BI__builtin_ia32_sqrtsh_round_mask:
  case X86::BI__builtin_ia32_sqrtsd_round_mask:
  case X86::BI__builtin_ia32_sqrtss_round_mask:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_sqrtph512:
  case X86::BI__builtin_ia32_sqrtps512:
  case X86::BI__builtin_ia32_sqrtpd512: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Value arg = ops[0];
    return cir::SqrtOp::create(builder, loc, arg.getType(), arg).getResult();
  }
  case X86::BI__builtin_ia32_pmuludq128:
  case X86::BI__builtin_ia32_pmuludq256:
  case X86::BI__builtin_ia32_pmuludq512: {
    unsigned opTypePrimitiveSizeInBits =
        cgm.getDataLayout().getTypeSizeInBits(ops[0].getType());
    return emitX86Muldq(builder, getLoc(expr->getExprLoc()), /*isSigned*/ false,
                        ops, opTypePrimitiveSizeInBits);
  }
  case X86::BI__builtin_ia32_pmuldq128:
  case X86::BI__builtin_ia32_pmuldq256:
  case X86::BI__builtin_ia32_pmuldq512: {
    unsigned opTypePrimitiveSizeInBits =
        cgm.getDataLayout().getTypeSizeInBits(ops[0].getType());
    return emitX86Muldq(builder, getLoc(expr->getExprLoc()), /*isSigned*/ true,
                        ops, opTypePrimitiveSizeInBits);
  }
  case X86::BI__builtin_ia32_pternlogd512_mask:
  case X86::BI__builtin_ia32_pternlogq512_mask:
  case X86::BI__builtin_ia32_pternlogd128_mask:
  case X86::BI__builtin_ia32_pternlogd256_mask:
  case X86::BI__builtin_ia32_pternlogq128_mask:
  case X86::BI__builtin_ia32_pternlogq256_mask:
  case X86::BI__builtin_ia32_pternlogd512_maskz:
  case X86::BI__builtin_ia32_pternlogq512_maskz:
  case X86::BI__builtin_ia32_pternlogd128_maskz:
  case X86::BI__builtin_ia32_pternlogd256_maskz:
  case X86::BI__builtin_ia32_pternlogq128_maskz:
  case X86::BI__builtin_ia32_pternlogq256_maskz:
  case X86::BI__builtin_ia32_vpshldd128:
  case X86::BI__builtin_ia32_vpshldd256:
  case X86::BI__builtin_ia32_vpshldd512:
  case X86::BI__builtin_ia32_vpshldq128:
  case X86::BI__builtin_ia32_vpshldq256:
  case X86::BI__builtin_ia32_vpshldq512:
  case X86::BI__builtin_ia32_vpshldw128:
  case X86::BI__builtin_ia32_vpshldw256:
  case X86::BI__builtin_ia32_vpshldw512:
  case X86::BI__builtin_ia32_vpshrdd128:
  case X86::BI__builtin_ia32_vpshrdd256:
  case X86::BI__builtin_ia32_vpshrdd512:
  case X86::BI__builtin_ia32_vpshrdq128:
  case X86::BI__builtin_ia32_vpshrdq256:
  case X86::BI__builtin_ia32_vpshrdq512:
  case X86::BI__builtin_ia32_vpshrdw128:
  case X86::BI__builtin_ia32_vpshrdw256:
  case X86::BI__builtin_ia32_vpshrdw512:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_reduce_fadd_pd512:
  case X86::BI__builtin_ia32_reduce_fadd_ps512:
  case X86::BI__builtin_ia32_reduce_fadd_ph512:
  case X86::BI__builtin_ia32_reduce_fadd_ph256:
  case X86::BI__builtin_ia32_reduce_fadd_ph128: {
    assert(!cir::MissingFeatures::fastMathFlags());
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "vector.reduce.fadd", ops[0].getType(),
                                       mlir::ValueRange{ops[0], ops[1]});
  }
  case X86::BI__builtin_ia32_reduce_fmul_pd512:
  case X86::BI__builtin_ia32_reduce_fmul_ps512:
  case X86::BI__builtin_ia32_reduce_fmul_ph512:
  case X86::BI__builtin_ia32_reduce_fmul_ph256:
  case X86::BI__builtin_ia32_reduce_fmul_ph128: {
    assert(!cir::MissingFeatures::fastMathFlags());
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "vector.reduce.fmul", ops[0].getType(),
                                       mlir::ValueRange{ops[0], ops[1]});
  }
  case X86::BI__builtin_ia32_reduce_fmax_pd512:
  case X86::BI__builtin_ia32_reduce_fmax_ps512:
  case X86::BI__builtin_ia32_reduce_fmax_ph512:
  case X86::BI__builtin_ia32_reduce_fmax_ph256:
  case X86::BI__builtin_ia32_reduce_fmax_ph128: {
    assert(!cir::MissingFeatures::fastMathFlags());
    cir::VectorType vecTy = cast<cir::VectorType>(ops[0].getType());
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "vector.reduce.fmax",
        vecTy.getElementType(), mlir::ValueRange{ops[0]});
  }
  case X86::BI__builtin_ia32_reduce_fmin_pd512:
  case X86::BI__builtin_ia32_reduce_fmin_ps512:
  case X86::BI__builtin_ia32_reduce_fmin_ph512:
  case X86::BI__builtin_ia32_reduce_fmin_ph256:
  case X86::BI__builtin_ia32_reduce_fmin_ph128: {
    assert(!cir::MissingFeatures::fastMathFlags());
    cir::VectorType vecTy = cast<cir::VectorType>(ops[0].getType());
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "vector.reduce.fmin",
        vecTy.getElementType(), mlir::ValueRange{ops[0]});
  }
  case X86::BI__builtin_ia32_rdrand16_step:
  case X86::BI__builtin_ia32_rdrand32_step:
  case X86::BI__builtin_ia32_rdrand64_step:
  case X86::BI__builtin_ia32_rdseed16_step:
  case X86::BI__builtin_ia32_rdseed32_step:
  case X86::BI__builtin_ia32_rdseed64_step: {
    llvm::StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_rdrand16_step:
      intrinsicName = "x86.rdrand.16";
      break;
    case X86::BI__builtin_ia32_rdrand32_step:
      intrinsicName = "x86.rdrand.32";
      break;
    case X86::BI__builtin_ia32_rdrand64_step:
      intrinsicName = "x86.rdrand.64";
      break;
    case X86::BI__builtin_ia32_rdseed16_step:
      intrinsicName = "x86.rdseed.16";
      break;
    case X86::BI__builtin_ia32_rdseed32_step:
      intrinsicName = "x86.rdseed.32";
      break;
    case X86::BI__builtin_ia32_rdseed64_step:
      intrinsicName = "x86.rdseed.64";
      break;
    }

    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Type randTy = cast<cir::PointerType>(ops[0].getType()).getPointee();
    llvm::SmallVector<mlir::Type, 2> resultTypes = {randTy,
                                                    builder.getUInt32Ty()};
    cir::RecordType resRecord =
        cir::RecordType::get(&getMLIRContext(), resultTypes, false, false,
                             cir::RecordType::RecordKind::Struct);

    mlir::Value call =
        builder.emitIntrinsicCallOp(loc, intrinsicName, resRecord);
    mlir::Value rand =
        cir::ExtractMemberOp::create(builder, loc, randTy, call, 0);
    builder.CIRBaseBuilderTy::createStore(loc, rand, ops[0]);

    return cir::ExtractMemberOp::create(builder, loc, builder.getUInt32Ty(),
                                        call, 1);
  }
  case X86::BI__builtin_ia32_addcarryx_u32:
  case X86::BI__builtin_ia32_addcarryx_u64:
  case X86::BI__builtin_ia32_subborrow_u32:
  case X86::BI__builtin_ia32_subborrow_u64:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_fpclassps128_mask:
  case X86::BI__builtin_ia32_fpclassps256_mask:
  case X86::BI__builtin_ia32_fpclassps512_mask:
  case X86::BI__builtin_ia32_vfpclassbf16128_mask:
  case X86::BI__builtin_ia32_vfpclassbf16256_mask:
  case X86::BI__builtin_ia32_vfpclassbf16512_mask:
  case X86::BI__builtin_ia32_fpclassph128_mask:
  case X86::BI__builtin_ia32_fpclassph256_mask:
  case X86::BI__builtin_ia32_fpclassph512_mask:
  case X86::BI__builtin_ia32_fpclasspd128_mask:
  case X86::BI__builtin_ia32_fpclasspd256_mask:
  case X86::BI__builtin_ia32_fpclasspd512_mask:
    return emitX86Fpclass(builder, getLoc(expr->getExprLoc()), builtinID, ops);
  case X86::BI__builtin_ia32_vp2intersect_q_512:
  case X86::BI__builtin_ia32_vp2intersect_q_256:
  case X86::BI__builtin_ia32_vp2intersect_q_128:
  case X86::BI__builtin_ia32_vp2intersect_d_512:
  case X86::BI__builtin_ia32_vp2intersect_d_256:
  case X86::BI__builtin_ia32_vp2intersect_d_128: {
    unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
    mlir::Location loc = getLoc(expr->getExprLoc());
    StringRef intrinsicName;

    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_vp2intersect_q_512:
      intrinsicName = "x86.avx512.vp2intersect.q.512";
      break;
    case X86::BI__builtin_ia32_vp2intersect_q_256:
      intrinsicName = "x86.avx512.vp2intersect.q.256";
      break;
    case X86::BI__builtin_ia32_vp2intersect_q_128:
      intrinsicName = "x86.avx512.vp2intersect.q.128";
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_512:
      intrinsicName = "x86.avx512.vp2intersect.d.512";
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_256:
      intrinsicName = "x86.avx512.vp2intersect.d.256";
      break;
    case X86::BI__builtin_ia32_vp2intersect_d_128:
      intrinsicName = "x86.avx512.vp2intersect.d.128";
      break;
    }

    auto resVector = cir::VectorType::get(builder.getBoolTy(), numElts);

    cir::RecordType resRecord =
        cir::RecordType::get(&getMLIRContext(), {resVector, resVector}, false,
                             false, cir::RecordType::RecordKind::Struct);

    mlir::Value call = builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), intrinsicName, resRecord,
        mlir::ValueRange{ops[0], ops[1]});
    mlir::Value result =
        cir::ExtractMemberOp::create(builder, loc, resVector, call, 0);
    result = emitX86MaskedCompareResult(builder, result, numElts, nullptr, loc);
    Address addr = Address(
        ops[2], clang::CharUnits::fromQuantity(std::max(1U, numElts / 8)));
    builder.createStore(loc, result, addr);

    result = cir::ExtractMemberOp::create(builder, loc, resVector, call, 1);
    result = emitX86MaskedCompareResult(builder, result, numElts, nullptr, loc);
    addr = Address(ops[3],
                   clang::CharUnits::fromQuantity(std::max(1U, numElts / 8)));
    builder.createStore(loc, result, addr);
    return mlir::Value{};
  }
  case X86::BI__builtin_ia32_vpmultishiftqb128:
  case X86::BI__builtin_ia32_vpmultishiftqb256:
  case X86::BI__builtin_ia32_vpmultishiftqb512:
  case X86::BI__builtin_ia32_vpshufbitqmb128_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb256_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb512_mask:
  case X86::BI__builtin_ia32_cmpeqps:
  case X86::BI__builtin_ia32_cmpeqpd:
  case X86::BI__builtin_ia32_cmpltps:
  case X86::BI__builtin_ia32_cmpltpd:
  case X86::BI__builtin_ia32_cmpleps:
  case X86::BI__builtin_ia32_cmplepd:
  case X86::BI__builtin_ia32_cmpunordps:
  case X86::BI__builtin_ia32_cmpunordpd:
  case X86::BI__builtin_ia32_cmpneqps:
  case X86::BI__builtin_ia32_cmpneqpd:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  case X86::BI__builtin_ia32_cmpnltps:
  case X86::BI__builtin_ia32_cmpnltpd:
    return emitVectorFCmp(builder, ops, getLoc(expr->getExprLoc()),
                          cir::CmpOpKind::lt, /*shouldInvert=*/true);
  case X86::BI__builtin_ia32_cmpnleps:
  case X86::BI__builtin_ia32_cmpnlepd:
    return emitVectorFCmp(builder, ops, getLoc(expr->getExprLoc()),
                          cir::CmpOpKind::le, /*shouldInvert=*/true);
  case X86::BI__builtin_ia32_cmpordps:
  case X86::BI__builtin_ia32_cmpordpd:
  case X86::BI__builtin_ia32_cmpph128_mask:
  case X86::BI__builtin_ia32_cmpph256_mask:
  case X86::BI__builtin_ia32_cmpph512_mask:
  case X86::BI__builtin_ia32_cmpps128_mask:
  case X86::BI__builtin_ia32_cmpps256_mask:
  case X86::BI__builtin_ia32_cmpps512_mask:
  case X86::BI__builtin_ia32_cmppd128_mask:
  case X86::BI__builtin_ia32_cmppd256_mask:
  case X86::BI__builtin_ia32_cmppd512_mask:
  case X86::BI__builtin_ia32_vcmpbf16512_mask:
  case X86::BI__builtin_ia32_vcmpbf16256_mask:
  case X86::BI__builtin_ia32_vcmpbf16128_mask:
  case X86::BI__builtin_ia32_cmpps:
  case X86::BI__builtin_ia32_cmpps256:
  case X86::BI__builtin_ia32_cmppd:
  case X86::BI__builtin_ia32_cmppd256:
  case X86::BI__builtin_ia32_cmpeqss:
  case X86::BI__builtin_ia32_cmpltss:
  case X86::BI__builtin_ia32_cmpless:
  case X86::BI__builtin_ia32_cmpunordss:
  case X86::BI__builtin_ia32_cmpneqss:
  case X86::BI__builtin_ia32_cmpnltss:
  case X86::BI__builtin_ia32_cmpnless:
  case X86::BI__builtin_ia32_cmpordss:
  case X86::BI__builtin_ia32_cmpeqsd:
  case X86::BI__builtin_ia32_cmpltsd:
  case X86::BI__builtin_ia32_cmplesd:
  case X86::BI__builtin_ia32_cmpunordsd:
  case X86::BI__builtin_ia32_cmpneqsd:
  case X86::BI__builtin_ia32_cmpnltsd:
  case X86::BI__builtin_ia32_cmpnlesd:
  case X86::BI__builtin_ia32_cmpordsd:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return {};
  case X86::BI__builtin_ia32_vcvtph2ps_mask:
  case X86::BI__builtin_ia32_vcvtph2ps256_mask:
  case X86::BI__builtin_ia32_vcvtph2ps512_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    return emitX86CvtF16ToFloatExpr(builder, loc, ops,
                                    convertType(expr->getType()));
  }
  case X86::BI__builtin_ia32_cvtneps2bf16_128_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    cir::VectorType resTy = cast<cir::VectorType>(convertType(expr->getType()));

    cir::VectorType inputTy = cast<cir::VectorType>(ops[0].getType());
    unsigned numElts = inputTy.getSize();

    mlir::Value mask = getMaskVecValue(builder, loc, ops[2], numElts);

    SmallVector<mlir::Value, 3> args;
    args.push_back(ops[0]);
    args.push_back(ops[1]);
    args.push_back(mask);

    return builder.emitIntrinsicCallOp(
        loc, "x86.avx512bf16.mask.cvtneps2bf16.128", resTy, args);
  }
  case X86::BI__builtin_ia32_cvtneps2bf16_256_mask:
  case X86::BI__builtin_ia32_cvtneps2bf16_512_mask: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    cir::VectorType resTy = cast<cir::VectorType>(convertType(expr->getType()));
    StringRef intrinsicName;
    if (builtinID == X86::BI__builtin_ia32_cvtneps2bf16_256_mask) {
      intrinsicName = "x86.avx512bf16.cvtneps2bf16.256";
    } else {
      assert(builtinID == X86::BI__builtin_ia32_cvtneps2bf16_512_mask);
      intrinsicName = "x86.avx512bf16.cvtneps2bf16.512";
    }

    mlir::Value res = builder.emitIntrinsicCallOp(loc, intrinsicName, resTy,
                                                  mlir::ValueRange{ops[0]});

    return emitX86Select(builder, loc, ops[2], res, ops[1]);
  }
  case X86::BI__cpuid:
  case X86::BI__cpuidex: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Value subFuncId = builtinID == X86::BI__cpuidex
                                ? ops[2]
                                : builder.getConstInt(loc, sInt32Ty, 0);
    cir::CpuIdOp::create(builder, loc, /*cpuInfo=*/ops[0],
                         /*functionId=*/ops[1], /*subFunctionId=*/subFuncId);
    return mlir::Value{};
  }
  case X86::BI__emul:
  case X86::BI__emulu:
  case X86::BI__mulh:
  case X86::BI__umulh:
  case X86::BI_mul128:
  case X86::BI_umul128: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  case X86::BI__faststorefence: {
    cir::AtomicFenceOp::create(
        builder, getLoc(expr->getExprLoc()),
        cir::MemOrder::SequentiallyConsistent,
        cir::SyncScopeKindAttr::get(&getMLIRContext(),
                                    cir::SyncScopeKind::System));
    return mlir::Value{};
  }
  case X86::BI__shiftleft128:
  case X86::BI__shiftright128: {
    // Flip low/high ops and zero-extend amount to matching type.
    // shiftleft128(Low, High, Amt) -> fshl(High, Low, Amt)
    // shiftright128(Low, High, Amt) -> fshr(High, Low, Amt)
    std::swap(ops[0], ops[1]);

    // Zero-extend shift amount to i64 if needed
    auto amtTy = mlir::cast<cir::IntType>(ops[2].getType());
    cir::IntType i64Ty = builder.getUInt64Ty();

    if (amtTy != i64Ty)
      ops[2] = builder.createIntCast(ops[2], i64Ty);

    const StringRef intrinsicName =
        (builtinID == X86::BI__shiftleft128) ? "fshl" : "fshr";
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), intrinsicName, i64Ty,
        mlir::ValueRange{ops[0], ops[1], ops[2]});
  }
  case X86::BI_ReadWriteBarrier:
  case X86::BI_ReadBarrier:
  case X86::BI_WriteBarrier: {
    cir::AtomicFenceOp::create(
        builder, getLoc(expr->getExprLoc()),
        cir::MemOrder::SequentiallyConsistent,
        cir::SyncScopeKindAttr::get(&getMLIRContext(),
                                    cir::SyncScopeKind::SingleThread));
    return mlir::Value{};
  }
  case X86::BI_AddressOfReturnAddress: {
    mlir::Location loc = getLoc(expr->getExprLoc());
    mlir::Value addr =
        cir::AddrOfReturnAddrOp::create(builder, loc, allocaInt8PtrTy);
    return builder.createCast(loc, cir::CastKind::bitcast, addr, voidPtrTy);
  }
  case X86::BI__stosb:
  case X86::BI__ud2:
  case X86::BI__int2c:
  case X86::BI__readfsbyte:
  case X86::BI__readfsword:
  case X86::BI__readfsdword:
  case X86::BI__readfsqword:
  case X86::BI__readgsbyte:
  case X86::BI__readgsword:
  case X86::BI__readgsdword:
  case X86::BI__readgsqword:
  case X86::BI__builtin_ia32_encodekey128_u32:
  case X86::BI__builtin_ia32_encodekey256_u32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
  case X86::BI__builtin_ia32_aesenc128kl_u8:
  case X86::BI__builtin_ia32_aesdec128kl_u8:
  case X86::BI__builtin_ia32_aesenc256kl_u8:
  case X86::BI__builtin_ia32_aesdec256kl_u8: {
    llvm::StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_aesenc128kl_u8:
      intrinsicName = "x86.aesenc128kl";
      break;
    case X86::BI__builtin_ia32_aesdec128kl_u8:
      intrinsicName = "x86.aesdec128kl";
      break;
    case X86::BI__builtin_ia32_aesenc256kl_u8:
      intrinsicName = "x86.aesenc256kl";
      break;
    case X86::BI__builtin_ia32_aesdec256kl_u8:
      intrinsicName = "x86.aesdec256kl";
      break;
    }

    return emitX86Aes(builder, getLoc(expr->getExprLoc()), intrinsicName,
                      convertType(expr->getType()), ops);
  }
  case X86::BI__builtin_ia32_aesencwide128kl_u8:
  case X86::BI__builtin_ia32_aesdecwide128kl_u8:
  case X86::BI__builtin_ia32_aesencwide256kl_u8:
  case X86::BI__builtin_ia32_aesdecwide256kl_u8: {
    llvm::StringRef intrinsicName;
    switch (builtinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_aesencwide128kl_u8:
      intrinsicName = "x86.aesencwide128kl";
      break;
    case X86::BI__builtin_ia32_aesdecwide128kl_u8:
      intrinsicName = "x86.aesdecwide128kl";
      break;
    case X86::BI__builtin_ia32_aesencwide256kl_u8:
      intrinsicName = "x86.aesencwide256kl";
      break;
    case X86::BI__builtin_ia32_aesdecwide256kl_u8:
      intrinsicName = "x86.aesdecwide256kl";
      break;
    }

    return emitX86Aeswide(builder, getLoc(expr->getExprLoc()), intrinsicName,
                          convertType(expr->getType()), ops);
  }
  case X86::BI__builtin_ia32_vfcmaddcph512_mask:
  case X86::BI__builtin_ia32_vfmaddcph512_mask:
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask:
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask:
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask3:
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask3:
  case X86::BI__builtin_ia32_prefetchi:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented X86 builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinID));
    return mlir::Value{};
  }
}
