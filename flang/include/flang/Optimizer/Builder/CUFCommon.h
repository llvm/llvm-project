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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"

static constexpr llvm::StringRef cudaDeviceModuleName = "cuda_device_mod";
static constexpr llvm::StringRef cudaSharedMemSuffix = "__shared_mem";

namespace fir {
class FirOpBuilder;
class KindMapping;
} // namespace fir

namespace cuf {

/// Retrieve or create the CUDA Fortran GPU module in the given \p mod.
mlir::gpu::GPUModuleOp getOrCreateGPUModule(mlir::ModuleOp mod,
                                            mlir::SymbolTable &symTab);

bool isCUDADeviceContext(mlir::Operation *op);
bool isCUDADeviceContext(mlir::Region &,
                         bool isDoConcurrentOffloadEnabled = false);
bool isRegisteredDeviceGlobal(fir::GlobalOp op);
bool isRegisteredDeviceAttr(std::optional<cuf::DataAttribute> attr);

void genPointerSync(const mlir::Value box, fir::FirOpBuilder &builder);

int computeElementByteSize(mlir::Location loc, mlir::Type type,
                           fir::KindMapping &kindMap,
                           bool emitErrorOnFailure = true);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFCOMMON_H_
