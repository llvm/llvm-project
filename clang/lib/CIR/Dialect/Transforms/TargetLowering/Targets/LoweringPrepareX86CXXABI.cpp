//====- LoweringPrepareX86CXXABI.cpp - Arm64 ABI specific code -------====//
//
// Part of the LLVM Project,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// This file provides X86{_64, _32} C++ ABI specific code that is used during
// LLVMIR lowering prepare.
//
//===------------------------------------------------------------------===//

#include "../LowerModule.h"
#include "../LoweringPrepareItaniumCXXABI.h"
#include "ABIInfoImpl.h"
#include "X86_64ABIInfo.h"

using namespace clang;
using namespace cir;

namespace {
class LoweringPrepareX86CXXABI : public LoweringPrepareItaniumCXXABI {
  bool is64;

public:
  LoweringPrepareX86CXXABI(bool is64) : is64(is64) {}
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override {
    if (is64)
      return lowerVAArgX86_64(builder, op, datalayout);

    return lowerVAArgX86_32(builder, op, datalayout);
  }

  mlir::Value lowerVAArgX86_64(cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
                               const cir::CIRDataLayout &datalayout);
  mlir::Value lowerVAArgX86_32(cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
                               const cir::CIRDataLayout &datalayout) {
    llvm_unreachable("lowerVAArg for X86_32 not implemented yet");
  }
};

std::unique_ptr<cir::LowerModule> getLowerModule(cir::VAArgOp op) {
  mlir::ModuleOp mo = op->getParentOfType<mlir::ModuleOp>();
  if (!mo)
    return nullptr;
  mlir::PatternRewriter rewriter(mo.getContext());
  return cir::createLowerModule(mo, rewriter);
}

mlir::Value buildX86_64VAArgFromMemory(cir::CIRBaseBuilderTy &builder,
                                       const cir::CIRDataLayout &datalayout,
                                       mlir::Value valist, mlir::Type Ty,
                                       mlir::Location loc) {
  mlir::Value overflow_arg_area_p =
      builder.createGetMemberOp(loc, valist, "overflow_arg_area", 2);
  mlir::Value overflow_arg_area = builder.createLoad(loc, overflow_arg_area_p);

  // AMD64-ABI 3.5.7p5: Step 7. Align l->overflow_arg_area upwards to a 16
  // byte boundary if alignment needed by type exceeds 8 byte boundary.
  // It isn't stated explicitly in the standard, but in practice we use
  // alignment greater than 16 where necessary.
  unsigned alignment = datalayout.getABITypeAlign(Ty).value();
  if (alignment > 8)
    overflow_arg_area =
        emitRoundPointerUpToAlignment(builder, overflow_arg_area, alignment);

  // AMD64-ABI 3.5.7p5: Step 8. Fetch type from l->overflow_arg_area.
  mlir::Value res = overflow_arg_area;

  // AMD64-ABI 3.5.7p5: Step 9. Set l->overflow_arg_area to:
  // l->overflow_arg_area + sizeof(type).
  // AMD64-ABI 3.5.7p5: Step 10. Align l->overflow_arg_area upwards to
  // an 8 byte boundary.
  uint64_t sizeInBytes = datalayout.getTypeStoreSize(Ty).getFixedValue();
  mlir::Value stride = builder.getSignedInt(loc, ((sizeInBytes + 7) & ~7), 32);
  mlir::Value castedPtr =
      builder.createPtrBitcast(overflow_arg_area, builder.getSIntNTy(8));
  overflow_arg_area = builder.createPtrStride(loc, castedPtr, stride);
  builder.createStore(loc, overflow_arg_area, overflow_arg_area_p);

  return res;
}

mlir::Value LoweringPrepareX86CXXABI::lowerVAArgX86_64(
    cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
    const cir::CIRDataLayout &datalayout) {
  // FIXME: return early since X86_64ABIInfo::classify can't handle these types.
  // Let's hope LLVM's va_arg instruction can take care of it.
  // Remove this when X86_64ABIInfo::classify can take care of every type.
  if (!mlir::isa<VoidType, IntType, SingleType, DoubleType, BoolType,
                 StructType, LongDoubleType>(op.getType()))
    return nullptr;

  // Assume that va_list type is correct; should be pointer to LLVM type:
  // struct {
  //   i32 gp_offset;
  //   i32 fp_offset;
  //   i8* overflow_arg_area;
  //   i8* reg_save_area;
  // };
  unsigned neededInt, neededSSE;

  std::unique_ptr<cir::LowerModule> lowerModule = getLowerModule(op);
  if (!lowerModule)
    return nullptr;
  mlir::Type ty = op.getType();

  // FIXME: How should we access the X86AVXABILevel?
  X86_64ABIInfo abiInfo(lowerModule->getTypes(), X86AVXABILevel::None);
  ABIArgInfo ai = abiInfo.classifyArgumentType(
      ty, 0, neededInt, neededSSE, /*isNamedArg=*/false, /*IsRegCall=*/false);

  // Empty records are ignored for parameter passing purposes.
  if (ai.isIgnore())
    return nullptr;

  mlir::Location loc = op.getLoc();
  mlir::Value valist = op.getOperand();

  // AMD64-ABI 3.5.7p5: Step 1. Determine whether type may be passed
  // in the registers. If not go to step 7.
  if (!neededInt && !neededSSE)
    return builder.createLoad(
        loc, builder.createPtrBitcast(buildX86_64VAArgFromMemory(
                                          builder, datalayout, valist, ty, loc),
                                      ty));

  auto currentBlock = builder.getInsertionBlock();

  // AMD64-ABI 3.5.7p5: Step 2. Compute num_gp to hold the number of
  // general purpose registers needed to pass type and num_fp to hold
  // the number of floating point registers needed.

  // AMD64-ABI 3.5.7p5: Step 3. Verify whether arguments fit into
  // registers. In the case: l->gp_offset > 48 - num_gp * 8 or
  // l->fp_offset > 304 - num_fp * 16 go to step 7.
  //
  // NOTE: 304 is a typo, there are (6 * 8 + 8 * 16) = 176 bytes of
  // register save space).

  mlir::Value inRegs;
  mlir::Value gp_offset_p, fp_offset_p;
  mlir::Value gp_offset, fp_offset;

  if (neededInt) {
    gp_offset_p = builder.createGetMemberOp(loc, valist, "gp_offset", 0);
    gp_offset = builder.createLoad(loc, gp_offset_p);
    inRegs = builder.getUnsignedInt(loc, 48 - neededInt * 8, 32);
    inRegs = builder.createCompare(loc, cir::CmpOpKind::le, gp_offset, inRegs);
  }

  if (neededSSE) {
    fp_offset_p = builder.createGetMemberOp(loc, valist, "fp_offset", 1);
    fp_offset = builder.createLoad(loc, fp_offset_p);
    mlir::Value fitsInFP =
        builder.getUnsignedInt(loc, 176 - neededSSE * 16, 32);
    fitsInFP =
        builder.createCompare(loc, cir::CmpOpKind::le, fp_offset, fitsInFP);
    inRegs = inRegs ? builder.createAnd(inRegs, fitsInFP) : fitsInFP;
  }

  mlir::Block *contBlock = currentBlock->splitBlock(op);
  mlir::Block *inRegBlock = builder.createBlock(contBlock);
  mlir::Block *inMemBlock = builder.createBlock(contBlock);
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<BrCondOp>(loc, inRegs, inRegBlock, inMemBlock);

  // Emit code to load the value if it was passed in registers.
  builder.setInsertionPointToStart(inRegBlock);

  // AMD64-ABI 3.5.7p5: Step 4. Fetch type from l->reg_save_area with
  // an offset of l->gp_offset and/or l->fp_offset. This may require
  // copying to a temporary location in case the parameter is passed
  // in different register classes or requires an alignment greater
  // than 8 for general purpose registers and 16 for XMM registers.
  //
  // FIXME: This really results in shameful code when we end up needing to
  // collect arguments from different places; often what should result in a
  // simple assembling of a structure from scattered addresses has many more
  // loads than necessary. Can we clean this up?
  mlir::Value regSaveArea = builder.createLoad(
      loc, builder.createGetMemberOp(loc, valist, "reg_save_area", 3));
  mlir::Value regAddr;

  uint64_t tyAlign = datalayout.getABITypeAlign(ty).value();
  // The alignment of result address.
  uint64_t alignment = 0;
  if (neededInt && neededSSE) {
    // FIXME: Cleanup.
    assert(ai.isDirect() && "Unexpected ABI info for mixed regs");
    StructType structTy = mlir::cast<StructType>(ai.getCoerceToType());
    cir::PointerType addrTy = builder.getPointerTo(ty);

    mlir::Value tmp = builder.createAlloca(loc, addrTy, ty, "tmp",
                                           CharUnits::fromQuantity(tyAlign));
    tmp = builder.createPtrBitcast(tmp, structTy);
    assert(structTy.getNumElements() == 2 &&
           "Unexpected ABI info for mixed regs");
    mlir::Type tyLo = structTy.getMembers()[0];
    mlir::Type tyHi = structTy.getMembers()[1];
    assert((isFPOrFPVectorTy(tyLo) ^ isFPOrFPVectorTy(tyHi)) &&
           "Unexpected ABI info for mixed regs");
    mlir::Value gpAddr = builder.createPtrStride(loc, regSaveArea, gp_offset);
    mlir::Value fpAddr = builder.createPtrStride(loc, regSaveArea, fp_offset);
    mlir::Value regLoAddr = isFPOrFPVectorTy(tyLo) ? fpAddr : gpAddr;
    mlir::Value regHiAddr = isFPOrFPVectorTy(tyHi) ? gpAddr : fpAddr;

    // Copy the first element.
    // FIXME: Our choice of alignment here and below is probably pessimistic.
    mlir::Value v = builder.createAlignedLoad(
        loc, regLoAddr, datalayout.getABITypeAlign(tyLo).value());
    builder.createStore(loc, v,
                        builder.createGetMemberOp(loc, tmp, "gp_offset", 0));

    // Copy the second element.
    v = builder.createAlignedLoad(loc, regHiAddr,
                                  datalayout.getABITypeAlign(tyHi).value());
    builder.createStore(loc, v,
                        builder.createGetMemberOp(loc, tmp, "fp_offset", 1));

    tmp = builder.createPtrBitcast(tmp, ty);
    regAddr = tmp;
  } else if (neededInt || neededSSE == 1) {
    uint64_t tySize = datalayout.getTypeStoreSize(ty).getFixedValue();

    mlir::Type coTy;
    if (ai.isDirect())
      coTy = ai.getCoerceToType();

    mlir::Value gpOrFpOffset = neededInt ? gp_offset : fp_offset;
    alignment = neededInt ? 8 : 16;
    uint64_t regSize = neededInt ? neededInt * 8 : 16;
    // There are two cases require special handling:
    // 1)
    //    ```
    //    struct {
    //      struct {} a[8];
    //      int b;
    //    };
    //    ```
    //    The lower 8 bytes of the structure are not stored,
    //    so an 8-byte offset is needed when accessing the structure.
    // 2)
    //   ```
    //   struct {
    //     long long a;
    //     struct {} b;
    //   };
    //   ```
    //   The stored size of this structure is smaller than its actual size,
    //   which may lead to reading past the end of the register save area.
    if (coTy && (ai.getDirectOffset() == 8 || regSize < tySize)) {
      cir::PointerType addrTy = builder.getPointerTo(ty);
      mlir::Value tmp = builder.createAlloca(loc, addrTy, ty, "tmp",
                                             CharUnits::fromQuantity(tyAlign));
      mlir::Value addr =
          builder.createPtrStride(loc, regSaveArea, gpOrFpOffset);
      mlir::Value src = builder.createAlignedLoad(
          loc, builder.createPtrBitcast(addr, coTy), tyAlign);
      mlir::Value ptrOffset =
          builder.getUnsignedInt(loc, ai.getDirectOffset(), 32);
      mlir::Value dst = builder.createPtrStride(loc, tmp, ptrOffset);
      builder.createStore(loc, src, dst);
      regAddr = tmp;
    } else {
      regAddr = builder.createPtrStride(loc, regSaveArea, gpOrFpOffset);

      // Copy into a temporary if the type is more aligned than the
      // register save area.
      if (neededInt && tyAlign > 8) {
        cir::PointerType addrTy = builder.getPointerTo(ty);
        mlir::Value tmp = builder.createAlloca(
            loc, addrTy, ty, "tmp", CharUnits::fromQuantity(tyAlign));
        builder.createMemCpy(loc, tmp, regAddr,
                             builder.getUnsignedInt(loc, tySize, 32));
        regAddr = tmp;
      }
    }

  } else {
    assert(neededSSE == 2 && "Invalid number of needed registers!");
    // SSE registers are spaced 16 bytes apart in the register save
    // area, we need to collect the two eightbytes together.
    // The ABI isn't explicit about this, but it seems reasonable
    // to assume that the slots are 16-byte aligned, since the stack is
    // naturally 16-byte aligned and the prologue is expected to store
    // all the SSE registers to the RSA.

    mlir::Value regAddrLo =
        builder.createPtrStride(loc, regSaveArea, fp_offset);
    mlir::Value regAddrHi = builder.createPtrStride(
        loc, regAddrLo, builder.getUnsignedInt(loc, 16, /*numBits=*/32));

    mlir::MLIRContext *Context = abiInfo.getContext().getMLIRContext();
    StructType structTy =
        ai.canHaveCoerceToType()
            ? cast<StructType>(ai.getCoerceToType())
            : StructType::get(
                  Context, {DoubleType::get(Context), DoubleType::get(Context)},
                  /*packed=*/false, StructType::Struct);
    cir::PointerType addrTy = builder.getPointerTo(ty);
    mlir::Value tmp = builder.createAlloca(loc, addrTy, ty, "tmp",
                                           CharUnits::fromQuantity(tyAlign));
    tmp = builder.createPtrBitcast(tmp, structTy);
    mlir::Value v = builder.createLoad(
        loc, builder.createPtrBitcast(regAddrLo, structTy.getMembers()[0]));
    builder.createStore(loc, v, builder.createGetMemberOp(loc, tmp, "", 0));
    v = builder.createLoad(
        loc, builder.createPtrBitcast(regAddrHi, structTy.getMembers()[1]));
    builder.createStore(loc, v, builder.createGetMemberOp(loc, tmp, "", 1));

    tmp = builder.createPtrBitcast(tmp, ty);
    regAddr = tmp;
  }

  // AMD64-ABI 3.5.7p5: Step 5. Set:
  // l->gp_offset = l->gp_offset + num_gp * 8
  // l->fp_offset = l->fp_offset + num_fp * 16.
  if (neededInt) {
    mlir::Value offset = builder.getUnsignedInt(loc, neededInt * 8, 32);
    builder.createStore(loc, builder.createAdd(gp_offset, offset), gp_offset_p);
  }

  if (neededSSE) {
    mlir::Value offset = builder.getUnsignedInt(loc, neededSSE * 8, 32);
    builder.createStore(loc, builder.createAdd(fp_offset, offset), fp_offset_p);
  }

  builder.create<BrOp>(loc, mlir::ValueRange{regAddr}, contBlock);

  // Emit code to load the value if it was passed in memory.
  builder.setInsertionPointToStart(inMemBlock);
  mlir::Value memAddr =
      buildX86_64VAArgFromMemory(builder, datalayout, valist, ty, loc);
  builder.create<BrOp>(loc, mlir::ValueRange{memAddr}, contBlock);

  // Return the appropriate result.
  builder.setInsertionPointToStart(contBlock);
  mlir::Value res_addr = contBlock->addArgument(regAddr.getType(), loc);

  return alignment
             ? builder.createAlignedLoad(
                   loc, builder.createPtrBitcast(res_addr, ty), alignment)
             : builder.createLoad(loc, builder.createPtrBitcast(res_addr, ty));
}
} // namespace

cir::LoweringPrepareCXXABI *
cir::LoweringPrepareCXXABI::createX86ABI(bool is64Bit) {
  return new LoweringPrepareX86CXXABI(is64Bit);
}
