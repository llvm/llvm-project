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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

static constexpr llvm::StringRef cudaDeviceModuleName = "cuda_device_mod";
static constexpr llvm::StringRef cudaSharedMemSuffix = "__shared_mem__";
/// Appended to the name of the device copy of a procedure while it lives in
/// the host module, where it cannot share the symbol of the original. The dot
/// keeps it clear of any Fortran or C identifier.
static constexpr llvm::StringRef cudaDeviceCopySuffix = ".device";

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

/// True for procedures that have a device side: attributes(device), (global),
/// (grid_global) and (host,device). Unlike isCUDADeviceContext, host_device
/// counts, since its body is compiled for the device as well.
bool isDeviceProcedure(mlir::func::FuncOp funcOp);

/// The device code of a module, as gathered by collectDeviceCode.
struct DeviceCodeSet {
  /// Procedures with a device proc attribute (see isDeviceProcedure).
  llvm::SetVector<mlir::func::FuncOp> deviceFuncs;
  /// Procedures without one that device code reaches, through calls or
  /// procedure references, directly or through other such procedures. OpenACC
  /// routines are excluded, the OpenACC pipeline moves those to the device
  /// itself. Declarations are included: device code needs them too.
  llvm::SetVector<mlir::func::FuncOp> calledFromDevice;
  /// Device procedures referenced from a derived-type binding table, whose
  /// host symbol must be kept for the table to verify.
  llvm::SetVector<mlir::func::FuncOp> keepInModule;
};

/// Gather the device code of \p mod. A host_device procedure that already has
/// a device copy is host code and is not walked for callees. With
/// \p rejectDynamicDispatch, a type-bound call with dynamic dispatch in device
/// code is reported as not yet implemented.
DeviceCodeSet collectDeviceCode(mlir::ModuleOp mod, mlir::SymbolTable &symTab,
                                bool rejectDynamicDispatch = false);

/// Point every fir.call and fir.address_of under \p root whose symbol is a key
/// of \p map at the mapped symbol instead.
void remapProcedureSymbols(
    mlir::Operation *root,
    const llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> &map);

/// Record on \p func the allocation policy of device code: the policy in
/// effect for it with stack arrays disabled, since the device stack is far
/// smaller than the host one.
void setDeviceAllocationPolicy(mlir::Operation *func);

void genPointerSync(const mlir::Value box, fir::FirOpBuilder &builder);

int computeElementByteSize(mlir::Location loc, mlir::Type type,
                           fir::KindMapping &kindMap,
                           bool emitErrorOnFailure = true);

mlir::Value computeElementCount(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value shapeOperand,
                                mlir::Type seqType, mlir::Type targetType);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_CUFCOMMON_H_
