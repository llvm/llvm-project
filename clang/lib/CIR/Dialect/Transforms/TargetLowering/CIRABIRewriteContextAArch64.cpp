//===- CIRABIRewriteContextAArch64.cpp - AArch64 ABI rewrite context -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AArch64 rewrite context, which expands cir.va_arg through the AAPCS64
// __stack / __gr_top / __vr_top / __gr_offs / __vr_offs va_list.
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

namespace {

class AArch64ABIRewriteContext final : public CIRABIRewriteContext {
public:
  AArch64ABIRewriteContext(mlir::ModuleOp module, const mlir::DataLayout &dl,
                           bool enableNoundefAttrs, bool softFloat)
      : CIRABIRewriteContext(module, dl, enableNoundefAttrs),
        softFloat(softFloat) {}

protected:
  mlir::LogicalResult expandVAArg(cir::VAArgOp op, const ArgClassification &ac,
                                  mlir::OpBuilder &opBuilder) override;

private:
  /// Whether the module uses the soft-float ABI, which has no vector
  /// registers, so floating-point and vector values are read from the general
  /// register save area.
  bool softFloat;
};

/// The alignment AAPCS64 places a variadic argument at.  This is the
/// unadjusted alignment, which an `aligned` attribute on a record does not
/// raise.
uint64_t unadjustedArgAlign(mlir::Type ty, mlir::ModuleOp modOp,
                            const mlir::DataLayout &dl) {
  if (auto recTy = mlir::dyn_cast<cir::RecordType>(ty))
    if (auto layout = cir::tryGetRecordLayout(modOp, recTy.getName()))
      return layout.getUnadjustedAlign();
  return dl.getTypeABIAlignment(ty);
}

/// Whether AAPCS64 treats \p ty as an aggregate rather than a scalar.
bool isAAPCS64Aggregate(mlir::Type ty) {
  return mlir::isa<cir::RecordType, cir::ArrayType, cir::ComplexType>(ty);
}

} // namespace

mlir::LogicalResult AArch64ABIRewriteContext::expandVAArg(
    cir::VAArgOp op, const ArgClassification &ac, mlir::OpBuilder &opBuilder) {
  CIRBaseBuilderTy builder(opBuilder);
  mlir::Location loc = op.getLoc();
  mlir::Type resultTy = op.getType();
  mlir::Value valist = op.getArgList();

  switch (ac.kind) {
  case ArgKind::Direct:
  case ArgKind::Extend:
    if (ac.directOffset)
      return reportVAArgNYI(op, "a type read at an offset into its register");
    break;
  case ArgKind::Indirect:
    if (ac.byVal)
      return reportVAArgNYI(op, "a type passed byval");
    break;
  case ArgKind::IndirectAliased:
    return reportVAArgNYI(op, "a type passed byref");
  case ArgKind::Expand:
    return reportVAArgNYI(op, "a type expanded into its fields");
  case ArgKind::Ignore:
    llvm_unreachable("rewriteVAArg replaces an ignored va_arg");
  }

  auto vaListRecTy = mlir::dyn_cast<cir::RecordType>(
      mlir::cast<cir::PointerType>(valist.getType()).getPointee());
  if (!vaListRecTy || vaListRecTy.getNumElements() != 5) {
    return reportVAArgNYI(op, "a va_list that is not the five-field __stack / "
                              "__gr_top / __vr_top / __gr_offs / __vr_offs "
                              "cursor");
  }
  llvm::ArrayRef<mlir::Type> vaFields = vaListRecTy.getMembers();

  cir::IntType byteTy = builder.getUIntNTy(8);
  bool isIndirect = ac.kind == ArgKind::Indirect;
  bool isBigEndian = cirDataLayout.isBigEndian();
  bool isAggregate = isAAPCS64Aggregate(resultTy);
  uint64_t tySize = dl.getTypeAllocSize(resultTy).getFixedValue();
  uint64_t tyAlign = unadjustedArgAlign(resultTy, module, dl);

  // An indirect argument is passed as a pointer to a copy.  Otherwise the
  // coerced type says which registers carry the value, and an array coercion
  // spreads it over one register per element.
  mlir::Type baseTy = resultTy;
  if (isIndirect)
    baseTy = builder.getPointerTo(byteTy);
  else if (ac.coercedType)
    baseTy = ac.coercedType;
  unsigned numRegs = 1;
  if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(baseTy)) {
    baseTy = arrTy.getElementType();
    numRegs = arrTy.getSize();
  }
  bool isFPR =
      !softFloat && mlir::isa<cir::FPTypeInterface, cir::VectorType>(baseTy);
  bool isHFA = isFPR && isAggregate;

  // Each vector register save slot is 16 bytes, whatever part of it the value
  // fills.  General register slots are 8 bytes and packed.
  unsigned offsIdx = isFPR ? 4 : 3;
  unsigned topIdx = isFPR ? 2 : 1;
  llvm::StringRef offsName = isFPR ? "__vr_offs" : "__gr_offs";
  llvm::StringRef topName = isFPR ? "__vr_top" : "__gr_top";
  uint64_t regSize =
      isFPR ? 16 * numRegs : llvm::alignTo(isIndirect ? 8 : tySize, 8);
  uint64_t regSlotSize = isFPR ? 16 : 8;

  builder.setInsertionPoint(op);

  // The offset is negative, counting up to zero, while registers remain.
  // A value aligned to more than 8 bytes starts at an even general register.
  mlir::Value offsP = builder.createGetMember(
      loc, builder.getPointerTo(vaFields[offsIdx]), valist, offsName, offsIdx);
  llvm::Align offsAlign =
      llvm::commonAlignment(llvm::Align(dl.getTypeABIAlignment(vaListRecTy)),
                            vaListRecTy.getElementOffset(dl, offsIdx));
  mlir::Value offs =
      builder.createLoad(loc, offsP, /*isVolatile=*/false, offsAlign.value());
  mlir::Type offsTy = offs.getType();
  mlir::Value zero = builder.getConstantInt(loc, offsTy, 0);
  mlir::Value usingStack =
      builder.createCompare(loc, cir::CmpOpKind::ge, offs, zero);
  mlir::Value regOffs = offs;
  if (!isFPR && !isIndirect && tyAlign > 8) {
    regOffs = builder.createAdd(
        loc, offs, builder.getConstantInt(loc, offsTy, tyAlign - 1));
    regOffs = builder.createAnd(
        loc, regOffs,
        builder.getConstantInt(loc, offsTy, -static_cast<int64_t>(tyAlign)));
  }
  mlir::Value newOffs = builder.createAdd(
      loc, regOffs, builder.getConstantInt(loc, offsTy, regSize));

  // Once the offset is non-negative it is left unchanged.  Otherwise it is
  // advanced past this value even when the value does not fit, since an
  // argument passed on the stack also uses up the remaining registers.
  mlir::Value inRegs =
      cir::TernaryOp::create(
          builder, loc, usingStack,
          /*trueBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            cir::YieldOp::create(b, l, mlir::Value(b.getBool(false, l)));
          },
          /*falseBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            b.createStore(l, newOffs, offsP, /*isVolatile=*/false,
                          /*isNontemporal=*/false,
                          b.getAlignmentAttr(offsAlign));
            mlir::Value fits =
                b.createCompare(l, cir::CmpOpKind::le, newOffs, zero);
            cir::YieldOp::create(b, l, fits);
          })
          .getResult();

  // The members of an HFA in registers are 16 bytes apart, so they are copied
  // into a temporary where they are contiguous.
  bool copyHFA = isHFA && numRegs > 1;
  mlir::Value hfaTemp;
  uint64_t regAlign = regSlotSize;
  if (copyHFA) {
    regAlign = std::max<uint64_t>(tyAlign, dl.getTypeABIAlignment(baseTy));
    hfaTemp = builder.createAlloca(
        loc, builder.getPointerTo(cir::ArrayType::get(baseTy, numRegs)),
        "vaarg.hfa", clang::CharUnits::fromQuantity(regAlign));
  }

  // On a big-endian target a value smaller than its slot is at the high end
  // of the slot.  In a register slot this applies to scalars and to the
  // members of an HFA.  In a stack slot it applies only to scalars.
  uint64_t regBEOffset = 0;
  if (isBigEndian && copyHFA) {
    uint64_t baseSize = dl.getTypeAllocSize(baseTy).getFixedValue();
    if (baseSize < 16)
      regBEOffset = 16 - baseSize;
  } else if (isBigEndian && !isIndirect && (isHFA || !isAggregate) &&
             tySize < regSlotSize) {
    regBEOffset = regSlotSize - tySize;
    regAlign =
        llvm::commonAlignment(llvm::Align(regAlign), regBEOffset).value();
  }
  uint64_t stackSlotSize = 8;
  uint64_t stackAlign = std::max<uint64_t>(stackSlotSize, tyAlign);
  uint64_t stackBEOffset = 0;
  if (isBigEndian && !isAggregate && tySize < stackSlotSize) {
    stackBEOffset = stackSlotSize - tySize;
    stackAlign =
        llvm::commonAlignment(llvm::Align(stackAlign), stackBEOffset).value();
  }

  mlir::Value addr =
      cir::TernaryOp::create(
          builder, loc, inRegs,
          /*trueBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            mlir::Value regTop = b.createLoad(
                l, b.createGetMember(l, b.getPointerTo(vaFields[topIdx]),
                                     valist, topName, topIdx));
            mlir::Value regAddr = b.createPtrStride(
                l, b.createPtrBitcast(regTop, byteTy), regOffs);
            if (copyHFA) {
              mlir::Value elems =
                  b.createCast(l, cir::CastKind::array_to_ptrdecay, hfaTemp,
                               b.getPointerTo(baseTy));
              uint64_t baseSize = dl.getTypeAllocSize(baseTy).getFixedValue();
              for (unsigned i = 0; i < numRegs; ++i) {
                uint64_t srcOffset = 16 * i + regBEOffset;
                mlir::Value src = b.createPtrStride(
                    l, regAddr, b.getSignedInt(l, srcOffset, 64));
                mlir::Value elem = b.createAlignedLoad(
                    l, b.createPtrBitcast(src, baseTy),
                    llvm::commonAlignment(llvm::Align(16), srcOffset).value());
                mlir::Value dst =
                    b.createPtrStride(l, elems, b.getSignedInt(l, i, 64));
                b.createStore(l, elem, dst, /*isVolatile=*/false,
                              /*isNontemporal=*/false,
                              b.getAlignmentAttr(llvm::commonAlignment(
                                  llvm::Align(regAlign), i * baseSize)));
              }
              regAddr = b.createPtrBitcast(hfaTemp, byteTy);
            } else if (regBEOffset) {
              regAddr = b.createPtrStride(l, regAddr,
                                          b.getSignedInt(l, regBEOffset, 64));
            }
            cir::YieldOp::create(b, l, regAddr);
          },
          /*falseBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            mlir::Value stackP = b.createGetMember(
                l, b.getPointerTo(vaFields[0]), valist, "__stack", 0);
            mlir::Value stack =
                b.createPtrBitcast(b.createLoad(l, stackP), byteTy);
            if (!isIndirect && tyAlign > 8)
              stack = roundPointerUpToAlignment(b, l, stack, tyAlign);
            uint64_t stackSize = isIndirect
                                     ? stackSlotSize
                                     : llvm::alignTo(tySize, stackSlotSize);
            b.createStore(
                l,
                b.createPtrStride(l, stack, b.getSignedInt(l, stackSize, 64)),
                stackP);
            if (stackBEOffset)
              stack = b.createPtrStride(l, stack,
                                        b.getSignedInt(l, stackBEOffset, 64));
            cir::YieldOp::create(b, l, stack);
          })
          .getResult();

  uint64_t addrAlign = std::min(regAlign, stackAlign);
  mlir::Value resultAddr;
  uint64_t resultAlign;
  if (isIndirect) {
    mlir::Value ptrAddr =
        builder.createPtrBitcast(addr, builder.getPointerTo(resultTy));
    resultAddr = builder.createAlignedLoad(loc, ptrAddr, addrAlign);
    resultAlign = tyAlign;
  } else {
    resultAddr = builder.createPtrBitcast(addr, resultTy);
    resultAlign = addrAlign;
  }

  if (mlir::isa<cir::RecordType>(resultTy)) {
    SmallPtrSet<mlir::Operation *, 1> noExceptions;
    if (failed(rewireRecordUses(builder, loc, op.getResult(), resultAddr,
                                resultAlign, noExceptions)))
      return mlir::failure();
  } else {
    op.getResult().replaceAllUsesWith(
        builder.createAlignedLoad(loc, resultAddr, resultAlign));
  }
  op->erase();
  return mlir::success();
}

std::unique_ptr<CIRABIRewriteContext>
cir::createAArch64ABIRewriteContext(mlir::ModuleOp module,
                                    const mlir::DataLayout &dl,
                                    bool enableNoundefAttrs, bool softFloat) {
  return std::make_unique<AArch64ABIRewriteContext>(
      module, dl, enableNoundefAttrs, softFloat);
}
