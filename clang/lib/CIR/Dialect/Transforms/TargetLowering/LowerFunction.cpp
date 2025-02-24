//===--- LowerFunction.cpp - Lower CIR Function Code ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenFunction.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//
#include "LowerFunction.h"
#include "CIRToCIRArgMapping.h"
#include "LowerCall.h"
#include "LowerFunctionInfo.h"
#include "LowerModule.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/TypeEvaluationKind.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = cir::ABIArgInfo;

namespace cir {

namespace {

mlir::Value buildAddressAtOffset(LowerFunction &LF, mlir::Value addr,
                                 const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    cir_cconv_unreachable("NYI");
  }
  return addr;
}

mlir::Value createCoercedBitcast(mlir::Value Src, mlir::Type DestTy,
                                 LowerFunction &CGF) {
  auto destPtrTy = PointerType::get(CGF.getRewriter().getContext(), DestTy);

  if (auto load = mlir::dyn_cast<LoadOp>(Src.getDefiningOp()))
    return CGF.getRewriter().create<CastOp>(Src.getLoc(), destPtrTy,
                                            CastKind::bitcast, load.getAddr());

  return CGF.getRewriter().create<CastOp>(Src.getLoc(), destPtrTy,
                                          CastKind::bitcast, Src);
}

// FIXME(cir): Create a custom rewriter class to abstract this away.
mlir::Value createBitcast(mlir::Value Src, mlir::Type Ty, LowerFunction &LF) {
  return LF.getRewriter().create<CastOp>(Src.getLoc(), Ty, CastKind::bitcast,
                                         Src);
}

/// Given a struct pointer that we are accessing some number of bytes out of it,
/// try to gep into the struct to get at its inner goodness.  Dive as deep as
/// possible without entering an element with an in-memory size smaller than
/// DstSize.
mlir::Value enterStructPointerForCoercedAccess(mlir::Value SrcPtr,
                                               StructType SrcSTy,
                                               uint64_t DstSize,
                                               LowerFunction &CGF) {
  // We can't dive into a zero-element struct.
  if (SrcSTy.getNumElements() == 0)
    cir_cconv_unreachable("NYI");

  mlir::Type FirstElt = SrcSTy.getMembers()[0];

  if (SrcSTy.isUnion())
    FirstElt = SrcSTy.getLargestMember(CGF.LM.getDataLayout().layout);

  // If the first elt is at least as large as what we're looking for, or if the
  // first element is the same size as the whole struct, we can enter it. The
  // comparison must be made on the store size and not the alloca size. Using
  // the alloca size may overstate the size of the load.
  uint64_t FirstEltSize = CGF.LM.getDataLayout().getTypeStoreSize(FirstElt);
  if (FirstEltSize < DstSize &&
      FirstEltSize < CGF.LM.getDataLayout().getTypeStoreSize(SrcSTy))
    return SrcPtr;

  auto &rw = CGF.getRewriter();
  auto *ctxt = rw.getContext();
  auto ptrTy = PointerType::get(ctxt, FirstElt);
  if (mlir::isa<StructType>(SrcPtr.getType())) {
    auto addr = SrcPtr;
    if (auto load = mlir::dyn_cast<LoadOp>(SrcPtr.getDefiningOp()))
      addr = load.getAddr();
    cir_cconv_assert(mlir::isa<PointerType>(addr.getType()));
    // we can not use getMemberOp here since we need a pointer to the first
    // element. And in the case of unions we pick a type of the largest elt,
    // that may or may not be the first one. Thus, getMemberOp verification
    // may fail.
    auto cast = createBitcast(addr, ptrTy, CGF);
    SrcPtr = rw.create<LoadOp>(SrcPtr.getLoc(), cast);
  }

  if (auto sty = mlir::dyn_cast<StructType>(SrcPtr.getType()))
    return enterStructPointerForCoercedAccess(SrcPtr, sty, DstSize, CGF);

  return SrcPtr;
}

/// Convert a value Val to the specific Ty where both
/// are either integers or pointers.  This does a truncation of the value if it
/// is too large or a zero extension if it is too small.
///
/// This behaves as if the value were coerced through memory, so on big-endian
/// targets the high bits are preserved in a truncation, while little-endian
/// targets preserve the low bits.
static mlir::Value coerceIntOrPtrToIntOrPtr(mlir::Value val, mlir::Type typ,
                                            LowerFunction &CGF) {
  if (val.getType() == typ)
    return val;

  auto &bld = CGF.getRewriter();

  if (mlir::isa<PointerType>(val.getType())) {
    // If this is Pointer->Pointer avoid conversion to and from int.
    if (mlir::isa<PointerType>(typ))
      return bld.create<CastOp>(val.getLoc(), typ, CastKind::bitcast, val);

    // Convert the pointer to an integer so we can play with its width.
    val = bld.create<CastOp>(val.getLoc(), typ, CastKind::ptr_to_int, val);
  }

  auto dstIntTy = typ;
  if (mlir::isa<PointerType>(dstIntTy))
    cir_cconv_unreachable("NYI");

  if (val.getType() != dstIntTy) {
    const auto &layout = CGF.LM.getDataLayout();
    if (layout.isBigEndian()) {
      // Preserve the high bits on big-endian targets.
      // That is what memory coercion does.
      uint64_t srcSize = layout.getTypeSizeInBits(val.getType());
      uint64_t dstSize = layout.getTypeSizeInBits(dstIntTy);
      uint64_t diff = srcSize > dstSize ? srcSize - dstSize : dstSize - srcSize;
      auto loc = val.getLoc();
      if (srcSize > dstSize) {
        auto intAttr = IntAttr::get(val.getType(), diff);
        auto amount = bld.create<ConstantOp>(loc, intAttr);
        val = bld.create<ShiftOp>(loc, val.getType(), val, amount, false);
        val = bld.create<CastOp>(loc, dstIntTy, CastKind::integral, val);
      } else {
        val = bld.create<CastOp>(loc, dstIntTy, CastKind::integral, val);
        auto intAttr = IntAttr::get(val.getType(), diff);
        auto amount = bld.create<ConstantOp>(loc, intAttr);
        val = bld.create<ShiftOp>(loc, val.getType(), val, amount, true);
      }
    } else {
      // Little-endian targets preserve the low bits. No shifts required.
      val = bld.create<CastOp>(val.getLoc(), dstIntTy, CastKind::integral, val);
    }
  }

  if (mlir::isa<PointerType>(typ))
    val = bld.create<CastOp>(val.getLoc(), typ, CastKind::int_to_ptr, val);

  return val;
}

AllocaOp createTmpAlloca(LowerFunction &LF, mlir::Location loc, mlir::Type ty) {
  auto &rw = LF.getRewriter();
  auto *ctxt = rw.getContext();
  mlir::PatternRewriter::InsertionGuard guard(rw);

  // find function's entry block and use it to find a best place for alloca
  auto *blk = rw.getBlock();
  auto *op = blk->getParentOp();
  FuncOp fun = mlir::dyn_cast<FuncOp>(op);
  if (!fun)
    fun = op->getParentOfType<FuncOp>();
  auto &entry = fun.getBody().front();

  auto ip = CIRBaseBuilderTy::getBestAllocaInsertPoint(&entry);
  rw.restoreInsertionPoint(ip);

  auto align = LF.LM.getDataLayout().getABITypeAlign(ty);
  auto alignAttr = rw.getI64IntegerAttr(align.value());
  auto ptrTy = PointerType::get(ctxt, ty);
  return rw.create<AllocaOp>(loc, ptrTy, ty, "tmp", alignAttr);
}

bool isVoidPtr(mlir::Value v) {
  if (auto p = mlir::dyn_cast<PointerType>(v.getType()))
    return mlir::isa<VoidType>(p.getPointee());
  return false;
}

MemCpyOp createMemCpy(LowerFunction &LF, mlir::Value dst, mlir::Value src,
                      uint64_t len) {
  cir_cconv_assert(mlir::isa<PointerType>(src.getType()));
  cir_cconv_assert(mlir::isa<PointerType>(dst.getType()));

  auto *ctxt = LF.getRewriter().getContext();
  auto &rw = LF.getRewriter();
  auto voidPtr = PointerType::get(ctxt, cir::VoidType::get(ctxt));

  if (!isVoidPtr(src))
    src = createBitcast(src, voidPtr, LF);
  if (!isVoidPtr(dst))
    dst = createBitcast(dst, voidPtr, LF);

  auto i64Ty = IntType::get(ctxt, 64, false);
  auto length = rw.create<ConstantOp>(src.getLoc(), IntAttr::get(i64Ty, len));
  return rw.create<MemCpyOp>(src.getLoc(), dst, src, length);
}

cir::AllocaOp findAlloca(mlir::Operation *op) {
  if (!op)
    return {};

  if (auto al = mlir::dyn_cast<cir::AllocaOp>(op)) {
    return al;
  } else if (auto ret = mlir::dyn_cast<cir::ReturnOp>(op)) {
    auto vals = ret.getInput();
    if (vals.size() == 1)
      return findAlloca(vals[0].getDefiningOp());
  } else if (auto load = mlir::dyn_cast<cir::LoadOp>(op)) {
    return findAlloca(load.getAddr().getDefiningOp());
  } else if (auto cast = mlir::dyn_cast<cir::CastOp>(op)) {
    return findAlloca(cast.getSrc().getDefiningOp());
  }

  return {};
}

mlir::Value findAddr(mlir::Value v) {
  if (mlir::isa<cir::PointerType>(v.getType()))
    return v;

  auto op = v.getDefiningOp();
  if (!op || !mlir::isa<cir::CastOp, cir::LoadOp, cir::ReturnOp>(op))
    return {};

  return findAddr(op->getOperand(0));
}

/// Create a store to \param Dst from \param Src where the source and
/// destination may have different types.
///
/// This safely handles the case when the src type is larger than the
/// destination type; the upper bits of the src will be lost.
void createCoercedStore(mlir::Value Src, mlir::Value Dst, bool DstIsVolatile,
                        LowerFunction &CGF) {
  mlir::Type SrcTy = Src.getType();
  mlir::Type DstTy = Dst.getType();
  if (SrcTy == DstTy) {
    cir_cconv_unreachable("NYI");
  }

  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);
  auto dstPtrTy = mlir::dyn_cast<PointerType>(DstTy);

  if (dstPtrTy)
    if (auto dstSTy = mlir::dyn_cast<StructType>(dstPtrTy.getPointee()))
      if (SrcTy != dstSTy)
        Dst = enterStructPointerForCoercedAccess(Dst, dstSTy,
                                                 SrcSize.getFixedValue(), CGF);

  auto &layout = CGF.LM.getDataLayout();
  llvm::TypeSize DstSize = dstPtrTy
                               ? layout.getTypeAllocSize(dstPtrTy.getPointee())
                               : layout.getTypeAllocSize(DstTy);

  if (SrcSize.isScalable() || SrcSize <= DstSize) {
    if (mlir::isa<IntType>(SrcTy) && dstPtrTy &&
        mlir::isa<PointerType>(dstPtrTy.getPointee()) &&
        SrcSize == layout.getTypeAllocSize(dstPtrTy.getPointee())) {
      cir_cconv_unreachable("NYI");
    } else if (auto STy = mlir::dyn_cast<StructType>(SrcTy)) {
      cir_cconv_unreachable("NYI");
    } else {
      Dst = createCoercedBitcast(Dst, SrcTy, CGF);
      CGF.buildAggregateStore(Src, Dst, DstIsVolatile);
    }
  } else if (mlir::isa<IntType>(SrcTy)) {
    auto &bld = CGF.getRewriter();
    auto *ctxt = CGF.LM.getMLIRContext();
    auto dstIntTy = IntType::get(ctxt, DstSize.getFixedValue() * 8, false);
    Src = coerceIntOrPtrToIntOrPtr(Src, dstIntTy, CGF);
    auto ptrTy = PointerType::get(ctxt, dstIntTy);
    auto addr = bld.create<CastOp>(Dst.getLoc(), ptrTy, CastKind::bitcast, Dst);
    bld.create<StoreOp>(Dst.getLoc(), Src, addr);
  } else {
    auto tmp = createTmpAlloca(CGF, Src.getLoc(), SrcTy);
    CGF.getRewriter().create<StoreOp>(Src.getLoc(), Src, tmp);
    createMemCpy(CGF, Dst, tmp, DstSize.getFixedValue());
  }
}

/// Coerces a \param Src value to a value of type \param Ty.
///
/// This safely handles the case when the src type is smaller than the
/// destination type; in this situation the values of bits which not present in
/// the src are undefined.
///
/// NOTE(cir): This method has partial parity with CGCall's CreateCoercedLoad.
/// Unlike the original codegen, this function does not emit a coerced load
/// since CIR's type checker wouldn't allow it. Instead, it casts the existing
/// ABI-agnostic value to it's ABI-aware counterpart. Nevertheless, we should
/// try to follow the same logic as the original codegen for correctness.
mlir::Value createCoercedValue(mlir::Value Src, mlir::Type Ty,
                               LowerFunction &CGF) {
  mlir::Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, just reuse the exising load.
  if (SrcTy == Ty)
    return Src;

  // If it is the special boolean case, simply bitcast it.
  if ((mlir::isa<BoolType>(SrcTy) && mlir::isa<IntType>(Ty)) ||
      (mlir::isa<IntType>(SrcTy) && mlir::isa<BoolType>(Ty)))
    return createBitcast(Src, Ty, CGF);

  llvm::TypeSize DstSize = CGF.LM.getDataLayout().getTypeAllocSize(Ty);

  if (auto SrcSTy = mlir::dyn_cast<StructType>(SrcTy)) {
    Src = enterStructPointerForCoercedAccess(Src, SrcSTy,
                                             DstSize.getFixedValue(), CGF);
    SrcTy = Src.getType();
  }

  llvm::TypeSize SrcSize = CGF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((mlir::isa<IntType>(Ty) || mlir::isa<PointerType>(Ty)) &&
      (mlir::isa<IntType>(SrcTy) || mlir::isa<PointerType>(SrcTy))) {
    return coerceIntOrPtrToIntOrPtr(Src, Ty, CGF);
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    return CGF.buildAggregateBitcast(Src, Ty);
  }

  if (mlir::Value addr = findAddr(Src)) {
    auto tmpAlloca = createTmpAlloca(CGF, addr.getLoc(), Ty);
    createMemCpy(CGF, tmpAlloca, addr, SrcSize.getFixedValue());
    return CGF.getRewriter().create<LoadOp>(addr.getLoc(),
                                            tmpAlloca.getResult());
  }

  cir_cconv_unreachable("NYI");
}

mlir::Value emitAddressAtOffset(LowerFunction &LF, mlir::Value addr,
                                const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    cir_cconv_unreachable("NYI");
  }
  return addr;
}

/// Creates a coerced value from \param src having a type of \param ty which is
/// a non primitive type
mlir::Value createCoercedNonPrimitive(mlir::Value src, mlir::Type ty,
                                      LowerFunction &LF) {
  if (auto load = mlir::dyn_cast<LoadOp>(src.getDefiningOp())) {
    auto &bld = LF.getRewriter();
    auto addr = load.getAddr();

    auto oldAlloca = mlir::dyn_cast<AllocaOp>(addr.getDefiningOp());
    auto alloca = bld.create<AllocaOp>(
        src.getLoc(), bld.getType<PointerType>(ty), ty,
        /*name=*/llvm::StringRef(""), oldAlloca.getAlignmentAttr());

    auto tySize = LF.LM.getDataLayout().getTypeStoreSize(ty);
    createMemCpy(LF, alloca, addr, tySize.getFixedValue());
    auto newLoad = bld.create<LoadOp>(src.getLoc(), alloca.getResult());
    bld.replaceAllOpUsesWith(load, newLoad);

    return newLoad;
  }

  cir_cconv_unreachable("NYI");
}

/// After the calling convention is lowered, an ABI-agnostic type might have to
/// be loaded back to its ABI-aware couterpart so it may be returned. If they
/// differ, we have to do a coerced load. A coerced load, which means to load a
/// type to another despite that they represent the same value. The simplest
/// cases can be solved with a mere bitcast.
///
/// This partially replaces CreateCoercedLoad from the original codegen.
/// However, instead of emitting the load, it emits a cast.
///
/// FIXME(cir): Improve parity with the original codegen.
mlir::Value castReturnValue(mlir::Value Src, mlir::Type Ty, LowerFunction &LF) {
  mlir::Type SrcTy = Src.getType();

  // If SrcTy and Ty are the same, nothing to do.
  if (SrcTy == Ty)
    return Src;

  // If is the special boolean case, simply bitcast it.
  if (mlir::isa<BoolType>(SrcTy) && mlir::isa<IntType>(Ty))
    return createBitcast(Src, Ty, LF);

  auto intTy = mlir::dyn_cast<IntType>(Ty);
  if (intTy && !intTy.isPrimitive())
    return createCoercedNonPrimitive(Src, Ty, LF);

  llvm::TypeSize DstSize = LF.LM.getDataLayout().getTypeAllocSize(Ty);

  // FIXME(cir): Do we need the EnterStructPointerForCoercedAccess routine here?

  llvm::TypeSize SrcSize = LF.LM.getDataLayout().getTypeAllocSize(SrcTy);

  if ((mlir::isa<IntType>(Ty) || mlir::isa<PointerType>(Ty)) &&
      (mlir::isa<IntType>(SrcTy) || mlir::isa<PointerType>(SrcTy))) {
    cir_cconv_unreachable("NYI");
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    auto Cast = createCoercedBitcast(Src, Ty, LF);
    return LF.getRewriter().create<LoadOp>(Src.getLoc(), Cast);
  }

  // Otherwise do coercion through memory.
  if (auto addr = findAlloca(Src.getDefiningOp())) {
    auto &rewriter = LF.getRewriter();
    auto tmp = createTmpAlloca(LF, Src.getLoc(), Ty);
    createMemCpy(LF, tmp, addr, SrcSize.getFixedValue());
    return rewriter.create<LoadOp>(Src.getLoc(), tmp.getResult());
  }

  cir_cconv_unreachable("NYI");
}

} // namespace

// FIXME(cir): Pass SrcFn and NewFn around instead of having then as attributes.
LowerFunction::LowerFunction(LowerModule &LM, mlir::PatternRewriter &rewriter,
                             FuncOp srcFn, FuncOp newFn)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), NewFn(newFn),
      LM(LM) {}

LowerFunction::LowerFunction(LowerModule &LM, mlir::PatternRewriter &rewriter,
                             FuncOp srcFn, CallOp callOp)
    : Target(LM.getTarget()), rewriter(rewriter), SrcFn(srcFn), callOp(callOp),
      LM(LM) {}

/// This method has partial parity with CodeGenFunction::EmitFunctionProlog from
/// the original codegen. However, it focuses on the ABI-specific details. On
/// top of that, it is also responsible for rewriting the original function.
llvm::LogicalResult LowerFunction::buildFunctionProlog(
    const LowerFunctionInfo &FI, FuncOp Fn,
    llvm::MutableArrayRef<mlir::BlockArgument> Args) {
  // NOTE(cir): Skipping naked and implicit-return-zero functions here. These
  // are dealt with in CIRGen.

  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FI);
  cir_cconv_assert(Fn.getNumArguments() == IRFunctionArgs.totalIRArgs());

  // If we're using inalloca, all the memory arguments are GEPs off of the last
  // parameter, which is a pointer to the complete memory area.
  cir_cconv_assert(!cir::MissingFeatures::inallocaArgs());

  // Name the struct return parameter.
  cir_cconv_assert(!cir::MissingFeatures::sretArgs());

  // Track if we received the parameter as a pointer (indirect, byval, or
  // inalloca). If already have a pointer, EmitParmDecl doesn't need to copy it
  // into a local alloca for us.
  llvm::SmallVector<mlir::Value, 8> ArgVals;
  ArgVals.reserve(Args.size());

  // FIXME(cir): non-blocking workaround for argument types that are not yet
  // properly handled by the ABI.
  if (cirCConvAssertionMode && FI.arg_size() != Args.size()) {
    cir_cconv_assert(cir::MissingFeatures::ABIParameterCoercion());
    return llvm::success();
  }

  // Create a pointer value for every parameter declaration. This usually
  // entails copying one or more LLVM IR arguments into an alloca. Don't push
  // any cleanups or do anything that might unwind. We do that separately, so
  // we can push the cleanups in the correct order for the ABI.
  cir_cconv_assert(FI.arg_size() == Args.size());
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator info_it = FI.arg_begin();
  for (llvm::MutableArrayRef<mlir::BlockArgument>::const_iterator
           i = Args.begin(),
           e = Args.end();
       i != e; ++i, ++info_it, ++ArgNo) {
    const mlir::Value Arg = *i;
    const ABIArgInfo &ArgI = info_it->info;

    bool isPromoted = cir::MissingFeatures::varDeclIsKNRPromoted();
    // We are converting from ABIArgInfo type to VarDecl type directly, unless
    // the parameter is promoted. In this case we convert to
    // CGFunctionInfo::ArgInfo type with subsequent argument demotion.
    mlir::Type Ty = {};
    if (isPromoted)
      cir_cconv_unreachable("NYI");
    else
      Ty = Arg.getType();
    cir_cconv_assert(!cir::MissingFeatures::evaluationKind());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgI.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      auto AI = Fn.getArgument(FirstIRArg);
      mlir::Type LTy = Arg.getType();

      // Prepare parameter attributes. So far, only attributes for pointer
      // parameters are prepared. See
      // http://llvm.org/docs/LangRef.html#paramattrs.
      if (ArgI.getDirectOffset() == 0 && mlir::isa<PointerType>(LTy) &&
          mlir::isa<PointerType>(ArgI.getCoerceToType())) {
        cir_cconv_assert_or_abort(
            !cir::MissingFeatures::ABIPointerParameterAttrs(), "NYI");
      }

      // Prepare the argument value. If we have the trivial case, handle it
      // with no muss and fuss.
      if (!mlir::isa<StructType>(ArgI.getCoerceToType()) &&
          ArgI.getCoerceToType() == Ty && ArgI.getDirectOffset() == 0) {
        cir_cconv_assert(NumIRArgs == 1);

        // LLVM expects swifterror parameters to be used in very restricted
        // ways. Copy the value into a less-restricted temporary.
        mlir::Value V = AI;
        if (cir::MissingFeatures::extParamInfo()) {
          cir_cconv_unreachable("NYI");
        }

        // Ensure the argument is the correct type.
        if (V.getType() != ArgI.getCoerceToType())
          cir_cconv_unreachable("NYI");

        if (isPromoted)
          cir_cconv_unreachable("NYI");

        ArgVals.push_back(V);

        // NOTE(cir): Here we have a trivial case, which means we can just
        // replace all uses of the original argument with the new one.
        mlir::Value oldArg = SrcFn.getArgument(ArgNo);
        mlir::Value newArg = Fn.getArgument(FirstIRArg);
        rewriter.replaceAllUsesWith(oldArg, newArg);

        break;
      }

      cir_cconv_assert(!cir::MissingFeatures::vectorType());

      StructType STy = mlir::dyn_cast<StructType>(ArgI.getCoerceToType());
      if (ArgI.isDirect() && !ArgI.getCanBeFlattened() && STy &&
          STy.getNumElements() > 1) {
        cir_cconv_unreachable("NYI");
      }

      // Allocate original argument to be "uncoerced".
      // FIXME(cir): We should have a alloca op builder that does not required
      // the pointer type to be explicitly passed.
      // FIXME(cir): Get the original name of the argument, as well as the
      // proper alignment for the given type being allocated.
      auto Alloca = rewriter.create<AllocaOp>(
          Fn.getLoc(), rewriter.getType<PointerType>(Ty), Ty,
          /*name=*/llvm::StringRef(""),
          /*alignment=*/rewriter.getI64IntegerAttr(4));

      mlir::Value Ptr = buildAddressAtOffset(*this, Alloca.getResult(), ArgI);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      if (ArgI.isDirect() && ArgI.getCanBeFlattened() && STy &&
          STy.getNumElements() > 1) {
        auto ptrType = mlir::cast<PointerType>(Ptr.getType());
        llvm::TypeSize structSize =
            LM.getTypes().getDataLayout().getTypeAllocSize(STy);
        llvm::TypeSize ptrElementSize =
            LM.getTypes().getDataLayout().getTypeAllocSize(
                ptrType.getPointee());
        if (structSize.isScalable()) {
          cir_cconv_unreachable("NYI");
        } else {
          uint64_t srcSize = structSize.getFixedValue();
          uint64_t dstSize = ptrElementSize.getFixedValue();

          mlir::Value addrToStoreInto;
          if (srcSize <= dstSize) {
            addrToStoreInto = rewriter.create<CastOp>(
                Ptr.getLoc(), PointerType::get(STy, ptrType.getAddrSpace()),
                CastKind::bitcast, Ptr);
          } else {
            addrToStoreInto = createTmpAlloca(*this, Ptr.getLoc(), STy);
          }

          assert(STy.getNumElements() == NumIRArgs);
          for (unsigned i = 0, e = STy.getNumElements(); i != e; ++i) {
            mlir::Value ai = Fn.getArgument(FirstIRArg + i);
            mlir::Type elementTy = STy.getMembers()[i];
            mlir::Value eltPtr = rewriter.create<GetMemberOp>(
                ai.getLoc(),
                PointerType::get(elementTy, ptrType.getAddrSpace()),
                addrToStoreInto,
                /*name=*/"", /*index=*/i);
            rewriter.create<StoreOp>(ai.getLoc(), ai, eltPtr);
          }

          if (srcSize > dstSize) {
            createMemCpy(*this, Ptr, addrToStoreInto, dstSize);
          }
        }
      } else {
        // Simple case, just do a coerced store of the argument into the alloca.
        cir_cconv_assert(NumIRArgs == 1);
        mlir::Value AI = Fn.getArgument(FirstIRArg);
        // TODO(cir): Set argument name in the new function.
        createCoercedStore(AI, Ptr, /*DstIsVolatile=*/false, *this);
      }

      // Match to what EmitParamDecl is expecting for this type.
      if (cir::MissingFeatures::evaluationKind()) {
        cir_cconv_unreachable("NYI");
      } else {
        // FIXME(cir): Should we have an ParamValue abstraction like in the
        // original codegen?
        ArgVals.push_back(Alloca);
      }

      // NOTE(cir): Once we have uncoerced the argument, we should be able to
      // RAUW the original argument alloca with the new one. This assumes that
      // the argument is used only to be stored in a alloca.
      mlir::Value arg = SrcFn.getArgument(ArgNo);
      cir_cconv_assert(arg.hasOneUse());
      auto *firstStore = *arg.user_begin();
      auto argAlloca = mlir::cast<StoreOp>(firstStore).getAddr();
      rewriter.replaceAllUsesWith(argAlloca, Alloca);
      rewriter.eraseOp(firstStore);
      rewriter.eraseOp(argAlloca.getDefiningOp());
      break;
    }
    case ABIArgInfo::Indirect: {
      auto AI = Fn.getArgument(FirstIRArg);
      if (!hasScalarEvaluationKind(Ty)) {
        // Aggregates and complex variables are accessed by reference. All we
        // need to do is realign the value, if requested. Also, if the address
        // may be aliased, copy it to ensure that the parameter variable is
        // mutable and has a unique adress, as C requires.
        if (ArgI.getIndirectRealign() || ArgI.isIndirectAliased()) {
          cir_cconv_unreachable("NYI");
        } else {
          // Inspired by EmitParamDecl, which is called in the end of
          // EmitFunctionProlog in the original codegen
          cir_cconv_assert(!ArgI.getIndirectByVal() &&
                           "For truly ABI indirect arguments");

          auto ptrTy = rewriter.getType<PointerType>(Arg.getType());
          mlir::Value arg = SrcFn.getArgument(ArgNo);
          cir_cconv_assert(arg.hasOneUse());
          auto *firstStore = *arg.user_begin();
          auto argAlloca = mlir::cast<StoreOp>(firstStore).getAddr();

          rewriter.setInsertionPoint(argAlloca.getDefiningOp());
          auto align = LM.getDataLayout().getABITypeAlign(ptrTy);
          auto alignAttr = rewriter.getI64IntegerAttr(align.value());
          auto newAlloca = rewriter.create<AllocaOp>(
              Fn.getLoc(), rewriter.getType<PointerType>(ptrTy), ptrTy,
              /*name=*/llvm::StringRef(""),
              /*alignment=*/alignAttr);

          rewriter.create<StoreOp>(newAlloca.getLoc(), AI,
                                   newAlloca.getResult());
          auto load = rewriter.create<LoadOp>(newAlloca.getLoc(),
                                              newAlloca.getResult());

          rewriter.replaceAllUsesWith(argAlloca, load);
          rewriter.eraseOp(firstStore);
          rewriter.eraseOp(argAlloca.getDefiningOp());

          ArgVals.push_back(AI);
        }
      } else {
        cir_cconv_unreachable("NYI");
      }
      break;
    }
    default:
      cir_cconv_unreachable("Unhandled ABIArgInfo::Kind");
    }
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    cir_cconv_unreachable("NYI");
  } else {
    // FIXME(cir): In the original codegen, EmitParamDecl is called here. It
    // is likely that said function considers ABI details during emission, so
    // we migth have to add a counter part here. Currently, it is not needed.
  }

  return llvm::success();
}

llvm::LogicalResult
LowerFunction::buildFunctionEpilog(const LowerFunctionInfo &FI) {
  // NOTE(cir): no-return, naked, and no result functions should be handled in
  // CIRGen.

  mlir::Value RV = {};
  mlir::Type RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  switch (RetAI.getKind()) {

  case ABIArgInfo::Ignore:
    break;

  case ABIArgInfo::Indirect: {
    mlir::Value RVAddr = {};
    CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FI, true);
    if (IRFunctionArgs.hasSRetArg()) {
      auto &entry = NewFn.getBody().front();
      RVAddr = entry.getArgument(IRFunctionArgs.getSRetArgNo());
    }

    if (RVAddr) {
      mlir::PatternRewriter::InsertionGuard guard(rewriter);
      NewFn->walk([&](ReturnOp ret) {
        if (auto al = findAlloca(ret)) {
          rewriter.replaceAllUsesWith(al.getResult(), RVAddr);
          rewriter.eraseOp(al);
          rewriter.setInsertionPoint(ret);

          auto retInputs = ret.getInput();
          assert(retInputs.size() == 1 && "return should only have one input");
          if (auto load = mlir::dyn_cast<LoadOp>(retInputs[0].getDefiningOp()))
            if (load.getResult().use_empty())
              rewriter.eraseOp(load);

          rewriter.replaceOpWithNewOp<ReturnOp>(ret);
        }
      });
    }
    break;
  }

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    // FIXME(cir): Should we call ConvertType(RetTy) here?
    if (RetAI.getCoerceToType() == RetTy && RetAI.getDirectOffset() == 0) {
      // The internal return value temp always will have pointer-to-return-type
      // type, just do a load.

      // If there is a dominating store to ReturnValue, we can elide
      // the load, zap the store, and usually zap the alloca.
      // NOTE(cir): This seems like a premature optimization case. Skipping it.
      if (cir::MissingFeatures::returnValueDominatingStoreOptmiization()) {
        cir_cconv_unreachable("NYI");
      }
      // Otherwise, we have to do a simple load.
      else {
        // NOTE(cir): Nothing to do here. The codegen already emitted this load
        // for us and there is no casting necessary to conform to the ABI. The
        // zero-extension is enforced by the return value's attribute. Just
        // early exit.
        return llvm::success();
      }
    } else {
      // NOTE(cir): Unlike the original codegen, CIR may have multiple return
      // statements in the function body. We have to handle this here.
      mlir::PatternRewriter::InsertionGuard guard(rewriter);
      NewFn->walk([&](ReturnOp returnOp) {
        rewriter.setInsertionPoint(returnOp);
        RV = castReturnValue(returnOp->getOperand(0), RetAI.getCoerceToType(),
                             *this);
        rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, RV);
      });
    }

    // TODO(cir): Should AutoreleaseResult be handled here?
    break;

  default:
    cir_cconv_unreachable("Unhandled ABIArgInfo::Kind");
  }

  return llvm::success();
}

/// Generate code for a function based on the ABI-specific information.
///
/// This method has partial parity with CodeGenFunction::GenerateCode, but it
/// focuses on the ABI-specific details. So a lot of codegen stuff is removed.
llvm::LogicalResult
LowerFunction::generateCode(FuncOp oldFn, FuncOp newFn,
                            const LowerFunctionInfo &FnInfo) {
  cir_cconv_assert(newFn && "generating code for null Function");
  auto Args = oldFn.getArguments();

  // Emit the ABI-specific function prologue.
  cir_cconv_assert(newFn.empty() && "Function already has a body");
  rewriter.setInsertionPointToEnd(newFn.addEntryBlock());
  if (buildFunctionProlog(FnInfo, newFn, oldFn.getArguments()).failed())
    return llvm::failure();

  // Ensure that old ABI-agnostic arguments uses were replaced.
  const auto hasNoUses = [](mlir::Value val) { return val.getUses().empty(); };
  cir_cconv_assert(std::all_of(Args.begin(), Args.end(), hasNoUses) &&
                   "Missing RAUW?");

  // NOTE(cir): While the new function has the ABI-aware parameters, the old
  // function still has the function logic. To complete the migration, we have
  // to move the old function body to the new function.

  // Backup references  to entry blocks.
  mlir::Block *srcBlock = &oldFn.getBody().front();
  mlir::Block *dstBlock = &newFn.getBody().front();

  // Ensure both blocks have the same number of arguments in order to
  // safely merge them.
  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), FnInfo, true);
  if (IRFunctionArgs.hasSRetArg()) {
    auto dstIndex = IRFunctionArgs.getSRetArgNo();
    auto retArg = dstBlock->getArguments()[dstIndex];
    srcBlock->insertArgument(dstIndex, retArg.getType(), retArg.getLoc());
  }

  // Migrate function body to new ABI-aware function.
  rewriter.inlineRegionBefore(oldFn.getBody(), newFn.getBody(),
                              newFn.getBody().end());

  // The block arguments of srcBlock are the old function's arguments. At this
  // point, all old arguments should be replaced with the lowered values.
  // Thus we could safely remove all the block arguments on srcBlock here.
  srcBlock->eraseArguments(0, srcBlock->getNumArguments());

  // Merge entry blocks to ensure correct branching.
  rewriter.mergeBlocks(srcBlock, dstBlock);

  // FIXME(cir): What about saving parameters for corotines? Should we do
  // something about it in this pass? If the change with the calling
  // convention, we might have to handle this here.

  // Emit the standard function epilogue.
  if (buildFunctionEpilog(FnInfo).failed())
    return llvm::failure();

  return llvm::success();
}

void LowerFunction::buildAggregateStore(mlir::Value Val, mlir::Value Dest,
                                        bool DestIsVolatile) {
  // In LLVM codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  cir_cconv_assert(mlir::isa<PointerType>(Dest.getType()) &&
                   "Storing in a non-pointer!");
  (void)DestIsVolatile;

  // Circumvent CIR's type checking.
  mlir::Type pointeeTy = mlir::cast<PointerType>(Dest.getType()).getPointee();
  if (Val.getType() != pointeeTy) {
    // NOTE(cir):  We only bitcast and store if the types have the same size.
    cir_cconv_assert((LM.getDataLayout().getTypeSizeInBits(Val.getType()) ==
                      LM.getDataLayout().getTypeSizeInBits(pointeeTy)) &&
                     "Incompatible types");
    auto loc = Val.getLoc();
    Val = rewriter.create<CastOp>(loc, pointeeTy, CastKind::bitcast, Val);
  }

  rewriter.create<StoreOp>(Val.getLoc(), Val, Dest);
}

mlir::Value LowerFunction::buildAggregateBitcast(mlir::Value Val,
                                                 mlir::Type DestTy) {
  auto Cast = createCoercedBitcast(Val, DestTy, *this);
  return rewriter.create<LoadOp>(Val.getLoc(), Cast);
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// FIXME(cir): This method has partial parity to CodeGenFunction's
/// EmitCallEpxr method defined in CGExpr.cpp. This could likely be
/// removed in favor of a more direct approach.
llvm::LogicalResult LowerFunction::rewriteCallOp(CallOp op,
                                                 ReturnValueSlot retValSlot) {

  // TODO(cir): Check if BlockCall, CXXMemberCall, CUDAKernelCall, or
  // CXXOperatorMember require special handling here. These should be handled
  // in CIRGen, unless there is call conv or ABI-specific stuff to be handled,
  // them we should do it here.

  // TODO(cir): Also check if Builtin and CXXPeseudoDtor need special handling
  // here. These should be handled in CIRGen, unless there is call conv or
  // ABI-specific stuff to be handled, them we should do it here.

  // NOTE(cir): There is no direct way to fetch the function type from the
  // CallOp, so we fetch it from the source function. This assumes the
  // function definition has not yet been lowered.

  FuncType fnType;
  if (SrcFn) {
    fnType = SrcFn.getFunctionType();
  } else if (op.isIndirect()) {
    if (auto ptrTy =
            mlir::dyn_cast<PointerType>(op.getIndirectCall().getType()))
      fnType = mlir::dyn_cast<FuncType>(ptrTy.getPointee());
  }

  cir_cconv_assert(fnType && "No callee function type");

  // Rewrite the call operation to abide to the ABI calling convention.
  auto Ret = rewriteCallOp(fnType, SrcFn, op, retValSlot);

  // Replace the original call result with the new one.
  if (Ret)
    rewriter.replaceAllUsesWith(op.getResult(), Ret);

  // Erase original ABI-agnostic call.
  rewriter.eraseOp(op);
  return llvm::success();
}

/// Rewrite a call operation to abide to the ABI calling convention.
///
/// FIXME(cir): This method has partial parity to CodeGenFunction's EmitCall
/// method defined in CGExpr.cpp. This could likely be removed in favor of a
/// more direct approach since most of the code here is exclusively CodeGen.
mlir::Value LowerFunction::rewriteCallOp(FuncType calleeTy, FuncOp origCallee,
                                         CallOp callOp,
                                         ReturnValueSlot retValSlot,
                                         mlir::Value Chain) {
  // NOTE(cir): Skip a bunch of function pointer stuff and AST declaration
  // asserts. Also skip sanitizers, as these should likely be handled at
  // CIRGen.
  CallArgList Args;
  if (Chain)
    cir_cconv_unreachable("NYI");

  // NOTE(cir): Call args were already emitted in CIRGen. Skip the evaluation
  // order done in CIRGen and just fetch the exiting arguments here.
  Args = callOp.getArgOperands();

  const LowerFunctionInfo &FnInfo = LM.getTypes().arrangeFreeFunctionCall(
      callOp.getArgOperands(), calleeTy, /*chainCall=*/false);

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type
  //   that does not include a prototype, [the default argument
  //   promotions are performed]. If the number of arguments does not
  //   equal the number of parameters, the behavior is undefined. If
  //   the function is defined with a type that includes a prototype,
  //   and either the prototype ends with an ellipsis (, ...) or the
  //   types of the arguments after promotion are not compatible with
  //   the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a
  //   prototype, and the types of the arguments after promotion are
  //   not compatible with those of the parameters after promotion,
  //   the behavior is undefined [except in some trivial cases].
  // That is, in the general case, we should assume that a call
  // through an unprototyped function type works like a *non-variadic*
  // call.  The way we make this work is to cast to the exact type
  // of the promoted arguments.
  //
  // Chain calls use this same code path to add the invisible chain parameter
  // to the function type.
  if ((origCallee && origCallee.getNoProto()) || Chain) {
    cir_cconv_assert_or_abort(cir::MissingFeatures::ABINoProtoFunctions(),
                              "NYI");
  }

  cir_cconv_assert(!cir::MissingFeatures::CUDA());

  // TODO(cir): LLVM IR has the concept of "CallBase", which is a base class
  // for all types of calls. Perhaps we should have a CIR interface to mimic
  // this class.
  CallOp CallOrInvoke = {};
  mlir::Value CallResult =
      rewriteCallOp(FnInfo, origCallee, callOp, retValSlot, Args, CallOrInvoke,
                    /*isMustTail=*/false, callOp.getLoc());

  // NOTE(cir): Skipping debug stuff here.

  return CallResult;
}

mlir::Value createAlloca(mlir::Location loc, mlir::Type type,
                         LowerFunction &CGF) {
  auto align = CGF.LM.getDataLayout().getABITypeAlign(type);
  auto alignAttr = CGF.getRewriter().getI64IntegerAttr(align.value());
  return CGF.getRewriter().create<AllocaOp>(
      loc, CGF.getRewriter().getType<PointerType>(type), type,
      /*name=*/llvm::StringRef(""), alignAttr);
}

// NOTE(cir): This method has partial parity to CodeGenFunction's EmitCall
// method in CGCall.cpp. When incrementing it, use the original codegen as a
// reference: add ABI-specific stuff and skip codegen stuff.
mlir::Value LowerFunction::rewriteCallOp(const LowerFunctionInfo &CallInfo,
                                         FuncOp Callee, CallOp Caller,
                                         ReturnValueSlot ReturnValue,
                                         CallArgList &CallArgs,
                                         CallOp CallOrInvoke, bool isMustTail,
                                         mlir::Location loc) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify.

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  mlir::Type RetTy = CallInfo.getReturnType(); // ABI-agnostic type.
  const cir::ABIArgInfo &RetAI = CallInfo.getReturnInfo();

  FuncType IRFuncTy = LM.getTypes().getFunctionType(CallInfo);

  // NOTE(cir): Some target/ABI related checks happen here. They are skipped
  // under the assumption that they are handled in CIRGen.

  // 1. Set up the arguments.

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  if (StructType ArgStruct = CallInfo.getArgStruct()) {
    cir_cconv_unreachable("NYI");
  }

  CIRToCIRArgMapping IRFunctionArgs(LM.getContext(), CallInfo);
  llvm::SmallVector<mlir::Value, 16> IRCallArgs(IRFunctionArgs.totalIRArgs());

  mlir::Value sRetPtr;
  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  if (RetAI.isIndirect() || RetAI.isCoerceAndExpand() || RetAI.isInAlloca()) {
    sRetPtr = createAlloca(loc, RetTy, *this);
    IRCallArgs[IRFunctionArgs.getSRetArgNo()] = sRetPtr;
  }

  cir_cconv_assert(!cir::MissingFeatures::swift());

  // NOTE(cir): Skipping lifetime markers here.

  // Translate all of the arguments as necessary to match the IR lowering.
  cir_cconv_assert(CallInfo.arg_size() == CallArgs.size() &&
                   "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (auto I = CallArgs.begin(), E = CallArgs.end(); I != E;
       ++I, ++info_it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = info_it->info;

    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      cir_cconv_unreachable("NYI");

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {

      if (mlir::isa<BoolType>(info_it->type)) {
        IRCallArgs[FirstIRArg] = *I;
        break;
      }

      if (!mlir::isa<StructType>(ArgInfo.getCoerceToType()) &&
          ArgInfo.getCoerceToType() == info_it->type &&
          ArgInfo.getDirectOffset() == 0) {
        cir_cconv_assert(NumIRArgs == 1);
        mlir::Value V;
        if (!mlir::isa<StructType>(I->getType())) {
          V = *I;
        } else {
          cir_cconv_unreachable("NYI");
        }

        if (cir::MissingFeatures::extParamInfo()) {
          cir_cconv_unreachable("NYI");
        }

        if (ArgInfo.getCoerceToType() != V.getType() &&
            mlir::isa<IntType>(V.getType()))
          cir_cconv_unreachable("NYI");

        if (FirstIRArg < IRFuncTy.getNumInputs() &&
            V.getType() != IRFuncTy.getInput(FirstIRArg))
          cir_cconv_unreachable("NYI");

        if (cir::MissingFeatures::undef())
          cir_cconv_unreachable("NYI");
        IRCallArgs[FirstIRArg] = V;
        break;
      }

      // FIXME: Avoid the conversion through memory if possible.
      mlir::Value Src = {};
      if (!mlir::isa<StructType>(I->getType())) {
        cir_cconv_unreachable("NYI");
      } else {
        // NOTE(cir): L/RValue stuff are left for CIRGen to handle.
        Src = *I;
      }

      // If the value is offst in memory, apply the offset now.
      // FIXME(cir): Is this offset already handled in CIRGen?
      Src = emitAddressAtOffset(*this, Src, ArgInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      // As an example, if we have SrcTy = struct { i32, i32, i32 }, then the
      // coerced type can be STy = struct { u64, i32 }. Hence a function with
      // a single argument SrcTy will be rewritten to take two arguments,
      // namely u64 and i32.
      StructType STy = mlir::dyn_cast<StructType>(ArgInfo.getCoerceToType());
      if (STy && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        mlir::Type SrcTy = Src.getType();
        llvm::TypeSize SrcTypeSize = LM.getDataLayout().getTypeAllocSize(SrcTy);
        llvm::TypeSize DstTypeSize = LM.getDataLayout().getTypeAllocSize(STy);

        if (SrcTypeSize.isScalable()) {
          cir_cconv_unreachable("NYI");
        } else {
          size_t SrcSize = SrcTypeSize.getFixedValue();
          size_t DstSize = DstTypeSize.getFixedValue();

          // Create a new temporary space and copy src in the front bits of it.
          // Other bits will be left untouched.
          // Note in OG, Src is of type Address, while here it is mlir::Value.
          // Here we need to first create another alloca to convert it into a
          // PointerType, so that we can call memcpy.
          if (SrcSize < DstSize) {
            auto Alloca = createTmpAlloca(*this, loc, STy);
            auto SrcAlloca = createTmpAlloca(*this, loc, SrcTy);
            rewriter.create<cir::StoreOp>(loc, Src, SrcAlloca);
            createMemCpy(*this, Alloca, SrcAlloca, SrcSize);
            Src = Alloca;
          } else {
            cir_cconv_unreachable("NYI");
          }

          assert(NumIRArgs == STy.getNumElements());
          for (unsigned I = 0; I != STy.getNumElements(); ++I) {
            mlir::Value Member = rewriter.create<cir::GetMemberOp>(
                loc, PointerType::get(STy.getMembers()[I]), Src, /*name=*/"",
                /*index=*/I);
            mlir::Value Load = rewriter.create<cir::LoadOp>(loc, Member);
            cir_cconv_assert(!cir::MissingFeatures::argHasMaybeUndefAttr());
            IRCallArgs[FirstIRArg + I] = Load;
          }
        }
      } else {
        // In the simple case, just pass the coerced loaded value.
        cir_cconv_assert(NumIRArgs == 1);
        mlir::Value Load =
            createCoercedValue(Src, ArgInfo.getCoerceToType(), *this);

        // FIXME(cir): We should probably handle CMSE non-secure calls here
        cir_cconv_assert(!cir::MissingFeatures::cmseNonSecureCallAttr());

        // since they are a ARM-specific feature.
        if (cir::MissingFeatures::undef())
          cir_cconv_unreachable("NYI");
        IRCallArgs[FirstIRArg] = Load;
      }

      break;
    }
    case ABIArgInfo::Indirect:
    case ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      // TODO(cir): For aggregate types
      // We want to avoid creating an unnecessary temporary+copy here;
      // however, we need one in three cases:
      // 1. If the argument is not byval, and we are required to copy the
      // 2. If the argument is byval, RV is not sufficiently aligned, and
      //    source.  (This case doesn't occur on any common architecture.)
      //    we cannot force it to be sufficiently aligned.
      // 3. If the argument is byval, but RV is not located in default
      //    or alloca address space.
      cir_cconv_assert(!::cir::MissingFeatures::skipTempCopy());

      mlir::Value alloca = findAlloca(I->getDefiningOp());

      // since they are a ARM-specific feature.
      if (::cir::MissingFeatures::undef())
        cir_cconv_unreachable("NYI");

      // TODO(cir): add check for cases where we don't need the memcpy
      auto tmpAlloca = createTmpAlloca(
          *this, alloca.getLoc(),
          mlir::cast<PointerType>(alloca.getType()).getPointee());
      auto tySize = LM.getDataLayout().getTypeAllocSize(I->getType());
      createMemCpy(*this, tmpAlloca, alloca, tySize.getFixedValue());
      IRCallArgs[FirstIRArg] = tmpAlloca;

      // NOTE(cir): Skipping Emissions, lifetime markers.

      break;
    }
    default:
      llvm::outs() << "Missing ABIArgInfo::Kind: " << ArgInfo.getKind() << "\n";
      cir_cconv_unreachable("NYI");
    }
  }

  // 2. Prepare the function pointer.
  // NOTE(cir): This is not needed for CIR.

  // 3. Perform the actual call.

  // NOTE(cir): CIRGen handle when to "deactive" cleanups. We also skip some
  // debugging stuff here.

  // Update the largest vector width if any arguments have vector types.
  cir_cconv_assert(!cir::MissingFeatures::vectorType());

  // Compute the calling convention and attributes.

  // FIXME(cir): Skipping call attributes for now. Not sure if we have to do
  // this at all since we already do it for the function definition.

  // FIXME(cir): Implement the required procedures for strictfp function and
  // fast-math.

  // FIXME(cir): Add missing call-site attributes here if they are
  // ABI/target-specific, otherwise, do it in CIRGen.

  // NOTE(cir): Deciding whether to use Call or Invoke is done in CIRGen.

  // Rewrite the actual call operation.
  // TODO(cir): Handle other types of CIR calls (e.g. cir.try_call).
  // NOTE(cir): We don't know if the callee was already lowered, so we only
  // fetch the name from the callee, while the return type is fetch from the
  // lowering types manager.

  CallOp newCallOp;

  if (Caller.isIndirect()) {
    rewriter.setInsertionPoint(Caller);
    auto val = Caller.getIndirectCall();
    auto ptrTy = PointerType::get(val.getContext(), IRFuncTy);
    auto callee =
        rewriter.create<CastOp>(val.getLoc(), ptrTy, CastKind::bitcast, val);
    newCallOp = rewriter.create<CallOp>(loc, callee, IRFuncTy, IRCallArgs);
  } else {
    newCallOp = rewriter.create<CallOp>(loc, Caller.getCalleeAttr(),
                                        IRFuncTy.getReturnType(), IRCallArgs);
  }

  auto extraAttrs =
      rewriter.getAttr<ExtraFuncAttributesAttr>(rewriter.getDictionaryAttr({}));
  newCallOp->setAttr("extra_attrs", extraAttrs);

  cir_cconv_assert(!cir::MissingFeatures::vectorType());

  // NOTE(cir): Skipping some ObjC, tail-call, debug, and attribute stuff
  // here.

  // 4. Finish the call.

  // NOTE(cir): Skipping no-return, isMustTail, swift error handling, and
  // writebacks here. These should be handled in CIRGen, I think.

  // Convert return value from ABI-agnostic to ABI-aware.
  mlir::Value Ret = [&] {
    // NOTE(cir): CIRGen already handled the emission of the return value. We
    // need only to handle the ABI-specific to ABI-agnostic cast here.
    switch (RetAI.getKind()) {

    case cir::ABIArgInfo::Ignore:
      // If we are ignoring an argument that had a result, make sure to
      // construct the appropriate return value for our caller.
      return getUndefRValue(RetTy);

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      mlir::Type RetIRTy = RetTy;
      if (RetAI.getCoerceToType() == RetIRTy && RetAI.getDirectOffset() == 0) {
        switch (getEvaluationKind(RetTy)) {
        case cir::TypeEvaluationKind::TEK_Scalar: {
          // If the argument doesn't match, perform a bitcast to coerce it.
          // This can happen due to trivial type mismatches. NOTE(cir):
          // Perhaps this section should handle CIR's boolean case.
          mlir::Value V = newCallOp.getResult();
          if (V.getType() != RetIRTy)
            cir_cconv_unreachable("NYI");
          return V;
        }
        default:
          cir_cconv_unreachable("NYI");
        }
      }

      // If coercing a fixed vector from a scalable vector for ABI
      // compatibility, and the types match, use the llvm.vector.extract
      // intrinsic to perform the conversion.
      if (cir::MissingFeatures::vectorType()) {
        cir_cconv_unreachable("NYI");
      }

      // FIXME(cir): Use return value slot here.
      mlir::Value RetVal = callOp.getResult();
      mlir::Value dstPtr;
      for (auto *user : Caller->getUsers()) {
        if (auto storeOp = mlir::dyn_cast<StoreOp>(user)) {
          assert(!dstPtr && "multiple destinations for the return value");
          dstPtr = storeOp.getAddr();
        }
      }

      // TODO(cir): Check for volatile return values.
      cir_cconv_assert(!cir::MissingFeatures::volatileTypes());

      // NOTE(cir): If the function returns, there should always be a valid
      // return value present. Instead of setting the return value here, we
      // should have the ReturnValueSlot object set it beforehand.
      if (!RetVal) {
        RetVal = callOp.getResult();
        // TODO(cir): Check for volatile return values.
        cir_cconv_assert(cir::MissingFeatures::volatileTypes());
      }

      // An empty record can overlap other data (if declared with
      // no_unique_address); omit the store for such types - as there is no
      // actual data to store.
      if (mlir::dyn_cast<StructType>(RetTy) &&
          mlir::cast<StructType>(RetTy).getNumElements() != 0) {
        RetVal = newCallOp.getResult();
        createCoercedStore(RetVal, dstPtr, false, *this);

        for (auto *user : Caller->getUsers())
          if (auto storeOp = mlir::dyn_cast<StoreOp>(user))
            rewriter.eraseOp(storeOp);
      }

      // NOTE(cir): No need to convert from a temp to an RValue. This is
      // done in CIRGen
      return RetVal;
    }
    case ABIArgInfo::Indirect: {
      auto load = rewriter.create<LoadOp>(loc, sRetPtr);
      return load.getResult();
    }
    default:
      llvm::errs() << "Unhandled ABIArgInfo kind: " << RetAI.getKind() << "\n";
      cir_cconv_unreachable("NYI");
    }
  }();

  // NOTE(cir): Skipping Emissions, lifetime markers, and dtors here that
  // should be handled in CIRGen.

  return Ret;
}

// NOTE(cir): This method has partial parity to CodeGenFunction's
// GetUndefRValue defined in CGExpr.cpp.
mlir::Value LowerFunction::getUndefRValue(mlir::Type Ty) {
  if (mlir::isa<VoidType>(Ty))
    return nullptr;

  llvm::outs() << "Missing undef handler for value type: " << Ty << "\n";
  cir_cconv_unreachable("NYI");
}

cir::TypeEvaluationKind LowerFunction::getEvaluationKind(mlir::Type type) {
  // FIXME(cir): Implement type classes for CIR types.
  if (mlir::isa<StructType>(type))
    return cir::TypeEvaluationKind::TEK_Aggregate;
  if (mlir::isa<BoolType, IntType, SingleType, DoubleType, LongDoubleType,
                VectorType, PointerType>(type))
    return cir::TypeEvaluationKind::TEK_Scalar;
  cir_cconv_unreachable("NYI");
}

} // namespace cir
