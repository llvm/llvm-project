//===- LowerToLLVMOffloadPrintf.h - Lower cir.offload.printf ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowerings of cir.offload.printf to the printf runtime interface of each
// device target.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOFFLOADPRINTF_H
#define CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOFFLOADPRINTF_H

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {
namespace direct {

/// Lowers \p op, whose lowered format and arguments are \p format and
/// \p args, to a call to vprintf.
mlir::LogicalResult lowerNVPTXPrintf(cir::OffloadPrintfOp op,
                                     mlir::Value format, mlir::ValueRange args,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     const mlir::DataLayout &dataLayout,
                                     mlir::SymbolTableCollection &symbolTables);

/// Lowers \p op, whose lowered operands are \p operands, to the buffered
/// printf runtime interface if \p isBuffered and to hostcall printf otherwise.
/// Strings are passed to the runtime in \p genericAddrSpace.
mlir::LogicalResult lowerAMDGPUPrintf(cir::OffloadPrintfOp op,
                                      mlir::ValueRange operands,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      const mlir::DataLayout &dataLayout,
                                      mlir::SymbolTableCollection &symbolTables,
                                      unsigned genericAddrSpace,
                                      bool isBuffered);

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOFFLOADPRINTF_H
