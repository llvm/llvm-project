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
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"

namespace fir {

struct CUDAIntrinsicLibrary : IntrinsicLibrary {

  // Constructors.
  explicit CUDAIntrinsicLibrary(fir::FirOpBuilder &builder, aiir::Location loc)
      : IntrinsicLibrary(builder, loc) {}
  CUDAIntrinsicLibrary() = delete;
  CUDAIntrinsicLibrary(const CUDAIntrinsicLibrary &) = delete;

  // CUDA intrinsic handlers.
  aiir::Value genAtomicAdd(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAtomicAddR2(aiir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  template <int extent>
  fir::ExtendedValue genAtomicAddVector(aiir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAtomicAddVector4x4(aiir::Type,
                                           llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAtomicAnd(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAtomicCas(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAtomicDec(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAtomicExch(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAtomicInc(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAtomicMax(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAtomicMin(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAtomicOr(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAtomicSub(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAtomicXor(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genBarrierArrive(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genBarrierArriveCnt(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genBarrierInit(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genBarrierTryWait(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genBarrierTryWaitSleep(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genClusterBlockIndex(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genClusterDimBlocks(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue
      genCUDASetDefaultStream(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue
      genCUDASetDefaultStreamArray(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue
      genCUDAGetDefaultStreamArg(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genCUDAGetDefaultStreamNull(aiir::Type,
                                          llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue
      genCUDAStreamSynchronize(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genCUDAStreamSynchronizeNull(aiir::Type,
                                           llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genCUDAStreamDestroy(aiir::Type,
                                          llvm::ArrayRef<fir::ExtendedValue>);
  void genFenceProxyAsync(llvm::ArrayRef<fir::ExtendedValue>);
  template <const char *fctName, int extent>
  fir::ExtendedValue genLDXXFunc(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genMatchAllSync(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genMatchAnySync(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <typename OpTy>
  aiir::Value genNVVMTime(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genSyncThreads(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genSyncThreadsAnd(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSyncThreadsCount(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSyncThreadsOr(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genSyncWarp(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genThisCluster(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genThisGrid(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genThisThreadBlock(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genThisWarp(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <aiir::NVVM::MemScopeKind scope>
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
  template <aiir::NVVM::VoteSyncKind kind>
  aiir::Value genVoteSync(aiir::Type, llvm::ArrayRef<aiir::Value>);
};

const IntrinsicHandler *findCUDAIntrinsicHandler(llvm::StringRef name);

} // namespace fir

#endif // FORTRAN_LOWER_CUDAINTRINSICCALL_H
