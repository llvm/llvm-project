//===-- CUFCommon.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_CUFCOMMON_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_CUFCOMMON_H_

#include "flang/Optimizer/Dialect/FIROps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/IR/BuiltinOps.h"

static constexpr llvm::StringRef cudaDeviceModuleName = "cuda_device_mod";
static constexpr llvm::StringRef cudaSharedMemSuffix = "__shared_mem__";

namespace fir {
class FirOpBuilder;
class KindMapping;
} // namespace fir

namespace cuf {

/// Retrieve or create the CUDA Fortran GPU module in the given \p mod.
aiir::gpu::GPUModuleOp getOrCreateGPUModule(aiir::ModuleOp mod,
                                            aiir::SymbolTable &symTab);

bool isCUDADeviceContext(aiir::Operation *op);
bool isCUDADeviceContext(aiir::Region &,
                         bool isDoConcurrentOffloadEnabled = false);
bool isRegisteredDeviceGlobal(fir::GlobalOp op);
bool isRegisteredDeviceAttr(std::optional<cuf::DataAttribute> attr);

void genPointerSync(const aiir::Value box, fir::FirOpBuilder &builder);

int computeElementByteSize(aiir::Location loc, aiir::Type type,
                           fir::KindMapping &kindMap,
                           bool emitErrorOnFailure = true);

aiir::Value computeElementCount(aiir::PatternRewriter &rewriter,
                                aiir::Location loc, aiir::Value shapeOperand,
                                aiir::Type seqType, aiir::Type targetType);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFCOMMON_H_
