//==-- Builder/CUDAIntrinsicCall.h - lowering of CUDA intrinsics ---*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CUDAINTRINSICCALL_H
#define FORTRAN_LOWER_CUDAINTRINSICCALL_H

#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace fir {

struct CUDAIntrinsicLibrary : IntrinsicLibrary {

  // Constructors.
  explicit CUDAIntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : IntrinsicLibrary(builder, loc) {}
  CUDAIntrinsicLibrary() = delete;
  CUDAIntrinsicLibrary(const CUDAIntrinsicLibrary &) = delete;

  // CUDA intrinsic handlers.
  mlir::Value genAtomicAdd(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAtomicAddR2(mlir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  template <int extent>
  fir::ExtendedValue genAtomicAddVector(mlir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAtomicAnd(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAtomicCas(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAtomicDec(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAtomicExch(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAtomicInc(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAtomicMax(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAtomicMin(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAtomicOr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAtomicSub(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAtomicXor(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genBarrierArrive(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genBarrierArriveCnt(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genBarrierInit(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genBarrierTryWait(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genBarrierTryWaitSleep(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genFenceProxyAsync(llvm::ArrayRef<fir::ExtendedValue>);
  template <const char *fctName, int extent>
  fir::ExtendedValue genLDXXFunc(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMatchAllSync(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genMatchAnySync(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <typename OpTy>
  mlir::Value genNVVMTime(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genSyncThreads(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSyncThreadsAnd(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSyncThreadsCount(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSyncThreadsOr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genSyncWarp(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genThisGrid(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genThisThreadBlock(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genThisWarp(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <mlir::NVVM::MemScopeKind scope>
  void genThreadFence(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkCommitGroup(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkG2S(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadC4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadC8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadI4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadI8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadR2(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadR4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkLoadR8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkS2G(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreC4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreC8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreI4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreI8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreR2(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreR4(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkStoreR8(llvm::ArrayRef<fir::ExtendedValue>);
  void genTMABulkWaitGroup(llvm::ArrayRef<fir::ExtendedValue>);
  template <mlir::NVVM::VoteSyncKind kind>
  mlir::Value genVoteSync(mlir::Type, llvm::ArrayRef<mlir::Value>);
};

const IntrinsicHandler *findCUDAIntrinsicHandler(llvm::StringRef name);

} // namespace fir

#endif // FORTRAN_LOWER_CUDAINTRINSICCALL_H
