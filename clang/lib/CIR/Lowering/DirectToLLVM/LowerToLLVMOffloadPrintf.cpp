//===- LowerToLLVMOffloadPrintf.cpp - Lower cir.offload.printf ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers cir.offload.printf to the printf runtime interface of each
// device target. The expansions follow the lowerings of the GPU dialect's
// gpu.printf in mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp.
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/TargetParser/Triple.h"

namespace cir {
namespace direct {

namespace {

// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   struct Tmp {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char* buf = alloca(sizeof(Tmp));
//   *(Tmp*)buf = {a1, a2, a3};
//   vprintf("format string", buf);
//
// `buf` is aligned to the max of {alignof(Arg1), ...}. Furthermore, each of
// the args is itself aligned to its ABI alignment.
//
// Note that by the time this function runs, the arguments have already
// undergone the standard C vararg promotion (short -> int, float -> double
// etc). In this function we pack the arguments into the buffer described above.
mlir::Value
packArgsIntoNVPTXFormatBuffer(cir::OffloadPrintfOp op, mlir::ValueRange args,
                              mlir::ConversionPatternRewriter &rewriter,
                              const mlir::DataLayout &dataLayout) {
  mlir::Location loc = op.getLoc();
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

  if (args.empty())
    // If there are no arguments other than the format string,
    // pass a nullptr to vprintf.
    return mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);

  // We can directly store the arguments into a struct, and the alignment
  // would automatically be correct. That's because vprintf does not
  // accept aggregates.
  auto allocaTy = mlir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), llvm::to_vector(args.getTypes()));

  // Allocate the buffer in the entry block, so that a printf in a loop does
  // not grow the stack on every iteration.
  mlir::Value alloca;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto fn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    rewriter.setInsertionPointToStart(&fn.getBody().front());
    mlir::Value one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), 1);
    alloca =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, allocaTy, one,
                                     dataLayout.getTypeABIAlignment(allocaTy));
  }

  // Member accesses are inbounds and nuw, as for cir.get_member.
  mlir::LLVM::GEPNoWrapFlags flags =
      mlir::LLVM::GEPNoWrapFlags::inbounds | mlir::LLVM::GEPNoWrapFlags::nuw;
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value member = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, allocaTy, alloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(i)}, flags);
    mlir::LLVM::StoreOp::create(rewriter, loc, arg, member,
                                dataLayout.getTypeABIAlignment(arg.getType()));
  }

  return alloca;
}

// Lowers a printf to a call to vprintf, as GPUPrintfOpToVPrintfLowering does
// for gpu.printf.
mlir::LogicalResult
lowerNVPTXPrintf(cir::OffloadPrintfOp op, mlir::Value format,
                 mlir::ValueRange args,
                 mlir::ConversionPatternRewriter &rewriter,
                 const mlir::DataLayout &dataLayout,
                 mlir::SymbolTableCollection &symbolTables) {
  mlir::Value packedData =
      packArgsIntoNVPTXFormatBuffer(op, args, rewriter, dataLayout);

  // int vprintf(char *format, void *packedData);
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Type i32Ty = rewriter.getI32Type();
  const llvm::StringRef fnName = "vprintf";
  createLLVMFuncOpIfNotExist(
      rewriter, symbolTables, op, fnName,
      mlir::LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy}));
  // vprintf takes a generic pointer, which can address a format string in any
  // address space.
  if (format.getType() != ptrTy)
    format = mlir::LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), ptrTy,
                                                 format);
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{i32Ty}, fnName, mlir::ValueRange{format, packedData});
  return mlir::success();
}

} // namespace

mlir::LogicalResult CIRToLLVMOffloadPrintfOpLowering::matchAndRewrite(
    cir::OffloadPrintfOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  llvm::StringRef tripleStr;
  if (auto tripleAttr = mod->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getTripleAttrName()))
    tripleStr = tripleAttr.getValue();
  llvm::Triple triple(tripleStr);

  if (triple.isNVPTX())
    return lowerNVPTXPrintf(op, adaptor.getFormat(), adaptor.getArgs(),
                            rewriter, dataLayout, symbolTables);

  return op.emitError() << "offload printf lowering is NYI for target '"
                        << tripleStr << "'";
}

} // namespace direct
} // namespace cir
