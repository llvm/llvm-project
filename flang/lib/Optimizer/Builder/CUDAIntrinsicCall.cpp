//===-- CUDAIntrinsicCall.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR for PowerPC
// intrinsics. Extensive use of MLIR interfaces and MLIR's coding style
// (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUDAIntrinsicCall.h"
#include "flang/Evaluate/common.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace fir {

using CI = CUDAIntrinsicLibrary;

static const char __ldca_i4x4[] = "__ldca_i4x4_";
static const char __ldca_i8x2[] = "__ldca_i8x2_";
static const char __ldca_r2x2[] = "__ldca_r2x2_";
static const char __ldca_r4x4[] = "__ldca_r4x4_";
static const char __ldca_r8x2[] = "__ldca_r8x2_";
static const char __ldcg_i4x4[] = "__ldcg_i4x4_";
static const char __ldcg_i8x2[] = "__ldcg_i8x2_";
static const char __ldcg_r2x2[] = "__ldcg_r2x2_";
static const char __ldcg_r4x4[] = "__ldcg_r4x4_";
static const char __ldcg_r8x2[] = "__ldcg_r8x2_";
static const char __ldcs_i4x4[] = "__ldcs_i4x4_";
static const char __ldcs_i8x2[] = "__ldcs_i8x2_";
static const char __ldcs_r2x2[] = "__ldcs_r2x2_";
static const char __ldcs_r4x4[] = "__ldcs_r4x4_";
static const char __ldcs_r8x2[] = "__ldcs_r8x2_";
static const char __ldcv_i4x4[] = "__ldcv_i4x4_";
static const char __ldcv_i8x2[] = "__ldcv_i8x2_";
static const char __ldcv_r2x2[] = "__ldcv_r2x2_";
static const char __ldcv_r4x4[] = "__ldcv_r4x4_";
static const char __ldcv_r8x2[] = "__ldcv_r8x2_";
static const char __ldlu_i4x4[] = "__ldlu_i4x4_";
static const char __ldlu_i8x2[] = "__ldlu_i8x2_";
static const char __ldlu_r2x2[] = "__ldlu_r2x2_";
static const char __ldlu_r4x4[] = "__ldlu_r4x4_";
static const char __ldlu_r8x2[] = "__ldlu_r8x2_";

// CUDA specific intrinsic handlers.
static constexpr IntrinsicHandler cudaHandlers[]{
    {"__ldca_i4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldca_i4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldca_i8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldca_i8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldca_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldca_r2x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldca_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldca_r4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldca_r8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldca_r8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcg_i4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcg_i4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcg_i8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcg_i8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcg_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcg_r2x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcg_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcg_r4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcg_r8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcg_r8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcs_i4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcs_i4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcs_i8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcs_i8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcs_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcs_r2x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcs_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcs_r4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcs_r8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcs_r8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcv_i4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcv_i4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcv_i8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcv_i8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcv_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcv_r2x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcv_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcv_r4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldcv_r8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldcv_r8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldlu_i4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldlu_i4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldlu_i8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldlu_i8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldlu_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldlu_r2x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldlu_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldlu_r4x4, 4>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"__ldlu_r8x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genLDXXFunc<__ldlu_r8x2, 2>),
     {{{"a", asAddr}}},
     /*isElemental=*/false},
    {"all_sync",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genVoteSync<mlir::NVVM::VoteSyncKind::all>),
     {{{"mask", asValue}, {"pred", asValue}}},
     /*isElemental=*/false},
    {"any_sync",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genVoteSync<mlir::NVVM::VoteSyncKind::any>),
     {{{"mask", asValue}, {"pred", asValue}}},
     /*isElemental=*/false},
    {"atomicadd_r4x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector<2>),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicadd_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector<4>),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicaddd",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicAdd),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicaddf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicAdd),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicaddi",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicAdd),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicaddl",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicAdd),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicaddr2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicAddR2),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicaddvector_r2x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector<2>),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicaddvector_r4x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector<2>),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicandi",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicAnd),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomiccasd",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicCas),
     {{{"a", asAddr}, {"v1", asValue}, {"v2", asValue}}},
     false},
    {"atomiccasf",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicCas),
     {{{"a", asAddr}, {"v1", asValue}, {"v2", asValue}}},
     false},
    {"atomiccasi",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicCas),
     {{{"a", asAddr}, {"v1", asValue}, {"v2", asValue}}},
     false},
    {"atomiccasul",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicCas),
     {{{"a", asAddr}, {"v1", asValue}, {"v2", asValue}}},
     false},
    {"atomicdeci",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicDec),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicexchd",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicExch),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicexchf",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicExch),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicexchi",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicExch),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicexchul",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicExch),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicinci",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicInc),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmaxd",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMax),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmaxf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMax),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmaxi",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMax),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmaxl",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMax),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmind",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMin),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicminf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMin),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicmini",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMin),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicminl",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicMin),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicori",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicOr),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicsubd",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicSub),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicsubf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicSub),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicsubi",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicSub),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicsubl",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genAtomicSub),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"atomicxori",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(&CI::genAtomicXor),
     {{{"a", asAddr}, {"v", asValue}}},
     false},
    {"ballot_sync",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genVoteSync<mlir::NVVM::VoteSyncKind::ballot>),
     {{{"mask", asValue}, {"pred", asValue}}},
     /*isElemental=*/false},
    {"barrier_arrive",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genBarrierArrive),
     {{{"barrier", asAddr}}},
     /*isElemental=*/false},
    {"barrier_arrive_cnt",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genBarrierArriveCnt),
     {{{"barrier", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"barrier_init",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genBarrierInit),
     {{{"barrier", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"barrier_try_wait",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genBarrierTryWait),
     {{{"barrier", asAddr}, {"token", asValue}}},
     /*isElemental=*/false},
    {"barrier_try_wait_sleep",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genBarrierTryWaitSleep),
     {{{"barrier", asAddr}, {"token", asValue}, {"ns", asValue}}},
     /*isElemental=*/false},
    {"clock",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genNVVMTime<mlir::NVVM::ClockOp>),
     {},
     /*isElemental=*/false},
    {"clock64",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genNVVMTime<mlir::NVVM::Clock64Op>),
     {},
     /*isElemental=*/false},
    {"fence_proxy_async",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genFenceProxyAsync),
     {},
     /*isElemental=*/false},
    {"globaltimer",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genNVVMTime<mlir::NVVM::GlobalTimerOp>),
     {},
     /*isElemental=*/false},
    {"match_all_syncjd",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAllSync),
     {{{"mask", asValue}, {"value", asValue}, {"pred", asAddr}}},
     /*isElemental=*/false},
    {"match_all_syncjf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAllSync),
     {{{"mask", asValue}, {"value", asValue}, {"pred", asAddr}}},
     /*isElemental=*/false},
    {"match_all_syncjj",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAllSync),
     {{{"mask", asValue}, {"value", asValue}, {"pred", asAddr}}},
     /*isElemental=*/false},
    {"match_all_syncjx",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAllSync),
     {{{"mask", asValue}, {"value", asValue}, {"pred", asAddr}}},
     /*isElemental=*/false},
    {"match_any_syncjd",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAnySync),
     {{{"mask", asValue}, {"value", asValue}}},
     /*isElemental=*/false},
    {"match_any_syncjf",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAnySync),
     {{{"mask", asValue}, {"value", asValue}}},
     /*isElemental=*/false},
    {"match_any_syncjj",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAnySync),
     {{{"mask", asValue}, {"value", asValue}}},
     /*isElemental=*/false},
    {"match_any_syncjx",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genMatchAnySync),
     {{{"mask", asValue}, {"value", asValue}}},
     /*isElemental=*/false},
    {"syncthreads",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genSyncThreads),
     {},
     /*isElemental=*/false},
    {"syncthreads_and_i4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsAnd),
     {},
     /*isElemental=*/false},
    {"syncthreads_and_l4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsAnd),
     {},
     /*isElemental=*/false},
    {"syncthreads_count_i4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsCount),
     {},
     /*isElemental=*/false},
    {"syncthreads_count_l4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsCount),
     {},
     /*isElemental=*/false},
    {"syncthreads_or_i4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsOr),
     {},
     /*isElemental=*/false},
    {"syncthreads_or_l4",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genSyncThreadsOr),
     {},
     /*isElemental=*/false},
    {"syncwarp",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(&CI::genSyncWarp),
     {},
     /*isElemental=*/false},
    {"this_grid",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genThisGrid),
     {},
     /*isElemental=*/false},
    {"this_thread_block",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genThisThreadBlock),
     {},
     /*isElemental=*/false},
    {"this_warp",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genThisWarp),
     {},
     /*isElemental=*/false},
    {"threadfence",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genThreadFence<mlir::NVVM::MemScopeKind::GPU>),
     {},
     /*isElemental=*/false},
    {"threadfence_block",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genThreadFence<mlir::NVVM::MemScopeKind::CTA>),
     {},
     /*isElemental=*/false},
    {"threadfence_system",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genThreadFence<mlir::NVVM::MemScopeKind::SYS>),
     {},
     /*isElemental=*/false},
    {"tma_bulk_commit_group",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkCommitGroup),
     {{}},
     /*isElemental=*/false},
    {"tma_bulk_g2s",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(&CI::genTMABulkG2S),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nbytes", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldc4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadC4),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldc8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadC8),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldi4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadI4),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldi8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadI8),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldr2",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadR2),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldr4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadR4),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_ldr8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkLoadR8),
     {{{"barrier", asAddr},
       {"src", asAddr},
       {"dst", asAddr},
       {"nelems", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_s2g",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(&CI::genTMABulkS2G),
     {{{"src", asAddr}, {"dst", asAddr}, {"nbytes", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_c4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreC4),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_c8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreC8),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_i4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreI4),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_i8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreI8),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_r2",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreR2),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_r4",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreR4),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_store_r8",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkStoreR8),
     {{{"src", asAddr}, {"dst", asAddr}, {"count", asValue}}},
     /*isElemental=*/false},
    {"tma_bulk_wait_group",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genTMABulkWaitGroup),
     {{}},
     /*isElemental=*/false},
};

template <std::size_t N>
static constexpr bool isSorted(const IntrinsicHandler (&array)[N]) {
  // Replace by std::sorted when C++20 is default (will be constexpr).
  const IntrinsicHandler *lastSeen{nullptr};
  bool isSorted{true};
  for (const auto &x : array) {
    if (lastSeen)
      isSorted &= std::string_view{lastSeen->name} < std::string_view{x.name};
    lastSeen = &x;
  }
  return isSorted;
}
static_assert(isSorted(cudaHandlers) && "map must be sorted");

const IntrinsicHandler *findCUDAIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &cudaHandler, llvm::StringRef name) {
    return name.compare(cudaHandler.name) > 0;
  };
  auto result = llvm::lower_bound(cudaHandlers, name, compare);
  return result != std::end(cudaHandlers) && result->name == name ? result
                                                                  : nullptr;
}

static mlir::Value convertPtrToNVVMSpace(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value barrier,
                                         mlir::NVVM::NVVMMemorySpace space) {
  mlir::Value llvmPtr = fir::ConvertOp::create(
      builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      barrier);
  mlir::Value addrCast = mlir::LLVM::AddrSpaceCastOp::create(
      builder, loc,
      mlir::LLVM::LLVMPointerType::get(builder.getContext(),
                                       static_cast<unsigned>(space)),
      llvmPtr);
  return addrCast;
}

static mlir::Value genAtomBinOp(fir::FirOpBuilder &builder, mlir::Location &loc,
                                mlir::LLVM::AtomicBinOp binOp, mlir::Value arg0,
                                mlir::Value arg1) {
  auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  arg0 = builder.createConvert(loc, llvmPointerType, arg0);
  return mlir::LLVM::AtomicRMWOp::create(builder, loc, binOp, arg0, arg1,
                                         mlir::LLVM::AtomicOrdering::seq_cst);
}

// ATOMICADD
mlir::Value
CUDAIntrinsicLibrary::genAtomicAdd(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  mlir::LLVM::AtomicBinOp binOp =
      mlir::isa<mlir::IntegerType>(args[1].getType())
          ? mlir::LLVM::AtomicBinOp::add
          : mlir::LLVM::AtomicBinOp::fadd;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicAddR2(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  mlir::Value a = fir::getBase(args[0]);

  if (mlir::isa<fir::BaseBoxType>(a.getType())) {
    a = fir::BoxAddrOp::create(builder, loc, a);
  }

  auto loc = builder.getUnknownLoc();
  auto f16Ty = builder.getF16Type();
  auto i32Ty = builder.getI32Type();
  auto vecF16Ty = mlir::VectorType::get({2}, f16Ty);
  mlir::Type idxTy = builder.getIndexType();
  auto f16RefTy = fir::ReferenceType::get(f16Ty);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto v1Coord = fir::CoordinateOp::create(builder, loc, f16RefTy,
                                           fir::getBase(args[1]), zero);
  auto v2Coord = fir::CoordinateOp::create(builder, loc, f16RefTy,
                                           fir::getBase(args[1]), one);
  auto v1 = fir::LoadOp::create(builder, loc, v1Coord);
  auto v2 = fir::LoadOp::create(builder, loc, v2Coord);
  mlir::Value undef = mlir::LLVM::UndefOp::create(builder, loc, vecF16Ty);
  mlir::Value vec1 = mlir::LLVM::InsertElementOp::create(
      builder, loc, undef, v1, builder.createIntegerConstant(loc, i32Ty, 0));
  mlir::Value vec2 = mlir::LLVM::InsertElementOp::create(
      builder, loc, vec1, v2, builder.createIntegerConstant(loc, i32Ty, 1));
  auto res = genAtomBinOp(builder, loc, mlir::LLVM::AtomicBinOp::fadd, a, vec2);
  auto i32VecTy = mlir::VectorType::get({1}, i32Ty);
  mlir::Value vecI32 =
      mlir::vector::BitCastOp::create(builder, loc, i32VecTy, res);
  return mlir::vector::ExtractOp::create(builder, loc, vecI32,
                                         mlir::ArrayRef<int64_t>{0});
}

// ATOMICADDVECTOR
template <int extent>
fir::ExtendedValue CUDAIntrinsicLibrary::genAtomicAddVector(
    mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value res = fir::AllocaOp::create(
      builder, loc, fir::SequenceType::get({extent}, resultType));
  mlir::Value a = fir::getBase(args[0]);
  if (mlir::isa<fir::BaseBoxType>(a.getType())) {
    a = fir::BoxAddrOp::create(builder, loc, a);
  }
  auto vecTy = mlir::VectorType::get({extent}, resultType);
  auto refTy = fir::ReferenceType::get(resultType);
  mlir::Type i32Ty = builder.getI32Type();
  mlir::Type idxTy = builder.getIndexType();

  // Extract the values from the array.
  llvm::SmallVector<mlir::Value> values;
  for (unsigned i = 0; i < extent; ++i) {
    mlir::Value pos = builder.createIntegerConstant(loc, idxTy, i);
    mlir::Value coord = fir::CoordinateOp::create(builder, loc, refTy,
                                                  fir::getBase(args[1]), pos);
    mlir::Value value = fir::LoadOp::create(builder, loc, coord);
    values.push_back(value);
  }
  // Pack extracted values into a vector to call the atomic add.
  mlir::Value undef = mlir::LLVM::UndefOp::create(builder, loc, vecTy);
  for (unsigned i = 0; i < extent; ++i) {
    mlir::Value insert = mlir::LLVM::InsertElementOp::create(
        builder, loc, undef, values[i],
        builder.createIntegerConstant(loc, i32Ty, i));
    undef = insert;
  }
  // Atomic operation with a vector of values.
  mlir::Value add =
      genAtomBinOp(builder, loc, mlir::LLVM::AtomicBinOp::fadd, a, undef);
  // Store results in the result array.
  for (unsigned i = 0; i < extent; ++i) {
    mlir::Value r = mlir::LLVM::ExtractElementOp::create(
        builder, loc, add, builder.createIntegerConstant(loc, i32Ty, i));
    mlir::Value c = fir::CoordinateOp::create(
        builder, loc, refTy, res, builder.createIntegerConstant(loc, idxTy, i));
    fir::StoreOp::create(builder, loc, r, c);
  }
  mlir::Value ext = builder.createIntegerConstant(loc, idxTy, extent);
  return fir::ArrayBoxValue(res, {ext});
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicAnd(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(mlir::isa<mlir::IntegerType>(args[1].getType()));

  mlir::LLVM::AtomicBinOp binOp = mlir::LLVM::AtomicBinOp::_and;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicOr(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(mlir::isa<mlir::IntegerType>(args[1].getType()));

  mlir::LLVM::AtomicBinOp binOp = mlir::LLVM::AtomicBinOp::_or;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICCAS
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicCas(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto successOrdering = mlir::LLVM::AtomicOrdering::acq_rel;
  auto failureOrdering = mlir::LLVM::AtomicOrdering::monotonic;
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(resultType.getContext());

  mlir::Value arg0 = fir::getBase(args[0]);
  mlir::Value arg1 = fir::getBase(args[1]);
  mlir::Value arg2 = fir::getBase(args[2]);

  auto bitCastFloat = [&](mlir::Value arg) -> mlir::Value {
    if (mlir::isa<mlir::Float32Type>(arg.getType()))
      return mlir::LLVM::BitcastOp::create(builder, loc, builder.getI32Type(),
                                           arg);
    if (mlir::isa<mlir::Float64Type>(arg.getType()))
      return mlir::LLVM::BitcastOp::create(builder, loc, builder.getI64Type(),
                                           arg);
    return arg;
  };

  arg1 = bitCastFloat(arg1);
  arg2 = bitCastFloat(arg2);

  if (arg1.getType() != arg2.getType()) {
    // arg1 and arg2 need to have the same type in AtomicCmpXchgOp.
    arg2 = builder.createConvert(loc, arg1.getType(), arg2);
  }

  auto address =
      mlir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy, arg0)
          .getResult(0);
  auto cmpxchg = mlir::LLVM::AtomicCmpXchgOp::create(
      builder, loc, address, arg1, arg2, successOrdering, failureOrdering);
  mlir::Value boolResult =
      mlir::LLVM::ExtractValueOp::create(builder, loc, cmpxchg, 1);
  return builder.createConvert(loc, resultType, boolResult);
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicDec(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(mlir::isa<mlir::IntegerType>(args[1].getType()));

  mlir::LLVM::AtomicBinOp binOp = mlir::LLVM::AtomicBinOp::udec_wrap;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICEXCH
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicExch(mlir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value arg0 = fir::getBase(args[0]);
  mlir::Value arg1 = fir::getBase(args[1]);
  assert(arg1.getType().isIntOrFloat());

  mlir::LLVM::AtomicBinOp binOp = mlir::LLVM::AtomicBinOp::xchg;
  return genAtomBinOp(builder, loc, binOp, arg0, arg1);
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicInc(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(mlir::isa<mlir::IntegerType>(args[1].getType()));

  mlir::LLVM::AtomicBinOp binOp = mlir::LLVM::AtomicBinOp::uinc_wrap;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicMax(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::LLVM::AtomicBinOp binOp =
      mlir::isa<mlir::IntegerType>(args[1].getType())
          ? mlir::LLVM::AtomicBinOp::max
          : mlir::LLVM::AtomicBinOp::fmax;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

mlir::Value
CUDAIntrinsicLibrary::genAtomicMin(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::LLVM::AtomicBinOp binOp =
      mlir::isa<mlir::IntegerType>(args[1].getType())
          ? mlir::LLVM::AtomicBinOp::min
          : mlir::LLVM::AtomicBinOp::fmin;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICSUB
mlir::Value
CUDAIntrinsicLibrary::genAtomicSub(mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  mlir::LLVM::AtomicBinOp binOp =
      mlir::isa<mlir::IntegerType>(args[1].getType())
          ? mlir::LLVM::AtomicBinOp::sub
          : mlir::LLVM::AtomicBinOp::fsub;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICXOR
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicXor(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value arg0 = fir::getBase(args[0]);
  mlir::Value arg1 = fir::getBase(args[1]);
  return genAtomBinOp(builder, loc, mlir::LLVM::AtomicBinOp::_xor, arg0, arg1);
}

// BARRIER_ARRIVE
mlir::Value
CUDAIntrinsicLibrary::genBarrierArrive(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  mlir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, args[0], mlir::NVVM::NVVMMemorySpace::Shared);
  return mlir::NVVM::MBarrierArriveOp::create(builder, loc, resultType, barrier)
      .getResult();
}

// BARRIER_ARRIBVE_CNT
mlir::Value
CUDAIntrinsicLibrary::genBarrierArriveCnt(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  mlir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, args[0], mlir::NVVM::NVVMMemorySpace::Shared);
  return mlir::NVVM::InlinePtxOp::create(builder, loc, {resultType},
                                         {barrier, args[1]}, {},
                                         "mbarrier.arrive.expect_tx.release."
                                         "cta.shared::cta.b64 %0, [%1], %2;",
                                         {})
      .getResult(0);
}

// BARRIER_INIT
void CUDAIntrinsicLibrary::genBarrierInit(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, fir::getBase(args[0]), mlir::NVVM::NVVMMemorySpace::Shared);
  mlir::NVVM::MBarrierInitOp::create(builder, loc, barrier,
                                     fir::getBase(args[1]), {});
  auto kind = mlir::NVVM::ProxyKindAttr::get(
      builder.getContext(), mlir::NVVM::ProxyKind::async_shared);
  auto space = mlir::NVVM::SharedSpaceAttr::get(
      builder.getContext(), mlir::NVVM::SharedSpace::shared_cta);
  mlir::NVVM::FenceProxyOp::create(builder, loc, kind, space);
}

// BARRIER_TRY_WAIT
mlir::Value
CUDAIntrinsicLibrary::genBarrierTryWait(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  mlir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  fir::StoreOp::create(builder, loc, zero, res);
  mlir::Value ns =
      builder.createIntegerConstant(loc, builder.getI32Type(), 1000000);
  mlir::Value load = fir::LoadOp::create(builder, loc, res);
  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{resultType}, mlir::ValueRange{load});
  mlir::Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
  mlir::Value beforeArg = beforeBlock->addArgument(resultType, loc);
  builder.setInsertionPointToStart(beforeBlock);
  mlir::Value condition = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, beforeArg, zero);
  mlir::scf::ConditionOp::create(builder, loc, condition, beforeArg);
  mlir::Block *afterBlock = builder.createBlock(&whileOp.getAfter());
  afterBlock->addArgument(resultType, loc);
  builder.setInsertionPointToStart(afterBlock);
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto barrier = builder.createConvert(loc, llvmPtrTy, args[0]);
  mlir::Value ret = mlir::NVVM::InlinePtxOp::create(
                        builder, loc, {resultType}, {barrier, args[1], ns}, {},
                        "{\n"
                        "  .reg .pred p;\n"
                        "  mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n"
                        "  selp.b32 %0, 1, 0, p;\n"
                        "}",
                        {})
                        .getResult(0);
  mlir::scf::YieldOp::create(builder, loc, ret);
  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(0);
}

// BARRIER_TRY_WAIT_SLEEP
mlir::Value
CUDAIntrinsicLibrary::genBarrierTryWaitSleep(mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto barrier = builder.createConvert(loc, llvmPtrTy, args[0]);
  return mlir::NVVM::InlinePtxOp::create(
             builder, loc, {resultType}, {barrier, args[1], args[2]}, {},
             "{\n"
             "  .reg .pred p;\n"
             "  mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n"
             "  selp.b32 %0, 1, 0, p;\n"
             "}",
             {})
      .getResult(0);
}

// FENCE_PROXY_ASYNC
void CUDAIntrinsicLibrary::genFenceProxyAsync(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  auto kind = mlir::NVVM::ProxyKindAttr::get(
      builder.getContext(), mlir::NVVM::ProxyKind::async_shared);
  auto space = mlir::NVVM::SharedSpaceAttr::get(
      builder.getContext(), mlir::NVVM::SharedSpace::shared_cta);
  mlir::NVVM::FenceProxyOp::create(builder, loc, kind, space);
}

// __LDCA, __LDCS, __LDLU, __LDCV
template <const char *fctName, int extent>
fir::ExtendedValue
CUDAIntrinsicLibrary::genLDXXFunc(mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  mlir::Type resTy = fir::SequenceType::get(extent, resultType);
  mlir::Value arg = fir::getBase(args[0]);
  mlir::Value res = fir::AllocaOp::create(builder, loc, resTy);
  if (mlir::isa<fir::BaseBoxType>(arg.getType()))
    arg = fir::BoxAddrOp::create(builder, loc, arg);
  mlir::Type refResTy = fir::ReferenceType::get(resTy);
  mlir::FunctionType ftype =
      mlir::FunctionType::get(arg.getContext(), {refResTy, refResTy}, {});
  auto funcOp = builder.createFunction(loc, fctName, ftype);
  llvm::SmallVector<mlir::Value> funcArgs;
  funcArgs.push_back(res);
  funcArgs.push_back(arg);
  fir::CallOp::create(builder, loc, funcOp, funcArgs);
  mlir::Value ext =
      builder.createIntegerConstant(loc, builder.getIndexType(), extent);
  return fir::ArrayBoxValue(res, {ext});
}

// CLOCK, CLOCK64, GLOBALTIMER
template <typename OpTy>
mlir::Value
CUDAIntrinsicLibrary::genNVVMTime(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 0 && "expect no arguments");
  return OpTy::create(builder, loc, resultType).getResult();
}

// MATCH_ALL_SYNC
mlir::Value
CUDAIntrinsicLibrary::genMatchAllSync(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);
  bool is32 = args[1].getType().isInteger(32) || args[1].getType().isF32();

  mlir::Type i1Ty = builder.getI1Type();
  mlir::MLIRContext *context = builder.getContext();

  mlir::Value arg1 = args[1];
  if (arg1.getType().isF32() || arg1.getType().isF64())
    arg1 = fir::ConvertOp::create(
        builder, loc, is32 ? builder.getI32Type() : builder.getI64Type(), arg1);

  mlir::Type retTy =
      mlir::LLVM::LLVMStructType::getLiteral(context, {resultType, i1Ty});
  auto match =
      mlir::NVVM::MatchSyncOp::create(builder, loc, retTy, args[0], arg1,
                                      mlir::NVVM::MatchSyncKind::all)
          .getResult();
  auto value = mlir::LLVM::ExtractValueOp::create(builder, loc, match, 0);
  auto pred = mlir::LLVM::ExtractValueOp::create(builder, loc, match, 1);
  auto conv = mlir::LLVM::ZExtOp::create(builder, loc, resultType, pred);
  fir::StoreOp::create(builder, loc, conv, args[2]);
  return value;
}

// MATCH_ANY_SYNC
mlir::Value
CUDAIntrinsicLibrary::genMatchAnySync(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  bool is32 = args[1].getType().isInteger(32) || args[1].getType().isF32();

  mlir::Value arg1 = args[1];
  if (arg1.getType().isF32() || arg1.getType().isF64())
    arg1 = fir::ConvertOp::create(
        builder, loc, is32 ? builder.getI32Type() : builder.getI64Type(), arg1);

  return mlir::NVVM::MatchSyncOp::create(builder, loc, resultType, args[0],
                                         arg1, mlir::NVVM::MatchSyncKind::any)
      .getResult();
}

// SYNCTHREADS
void CUDAIntrinsicLibrary::genSyncThreads(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  mlir::NVVM::Barrier0Op::create(builder, loc);
}

// SYNCTHREADS_AND
mlir::Value
CUDAIntrinsicLibrary::genSyncThreadsAnd(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  mlir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return mlir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             mlir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), mlir::NVVM::BarrierReduction::AND),
             arg)
      .getResult(0);
}

// SYNCTHREADS_COUNT
mlir::Value
CUDAIntrinsicLibrary::genSyncThreadsCount(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  mlir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return mlir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             mlir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), mlir::NVVM::BarrierReduction::POPC),
             arg)
      .getResult(0);
}

// SYNCTHREADS_OR
mlir::Value
CUDAIntrinsicLibrary::genSyncThreadsOr(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  mlir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return mlir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             mlir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), mlir::NVVM::BarrierReduction::OR),
             arg)
      .getResult(0);
}

// SYNCWARP
void CUDAIntrinsicLibrary::genSyncWarp(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  mlir::NVVM::SyncWarpOp::create(builder, loc, fir::getBase(args[0]));
}

// THIS_GRID
mlir::Value
CUDAIntrinsicLibrary::genThisGrid(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 0);
  auto recTy = mlir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  mlir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  mlir::Type i32Ty = builder.getI32Type();

  mlir::Value threadIdX = mlir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  mlir::Value threadIdY = mlir::NVVM::ThreadIdYOp::create(builder, loc, i32Ty);
  mlir::Value threadIdZ = mlir::NVVM::ThreadIdZOp::create(builder, loc, i32Ty);

  mlir::Value blockIdX = mlir::NVVM::BlockIdXOp::create(builder, loc, i32Ty);
  mlir::Value blockIdY = mlir::NVVM::BlockIdYOp::create(builder, loc, i32Ty);
  mlir::Value blockIdZ = mlir::NVVM::BlockIdZOp::create(builder, loc, i32Ty);

  mlir::Value blockDimX = mlir::NVVM::BlockDimXOp::create(builder, loc, i32Ty);
  mlir::Value blockDimY = mlir::NVVM::BlockDimYOp::create(builder, loc, i32Ty);
  mlir::Value blockDimZ = mlir::NVVM::BlockDimZOp::create(builder, loc, i32Ty);
  mlir::Value gridDimX = mlir::NVVM::GridDimXOp::create(builder, loc, i32Ty);
  mlir::Value gridDimY = mlir::NVVM::GridDimYOp::create(builder, loc, i32Ty);
  mlir::Value gridDimZ = mlir::NVVM::GridDimZOp::create(builder, loc, i32Ty);

  // this_grid.size = ((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y)) *
  // (blockDim.x * gridDim.x);
  mlir::Value resZ =
      mlir::arith::MulIOp::create(builder, loc, blockDimZ, gridDimZ);
  mlir::Value resY =
      mlir::arith::MulIOp::create(builder, loc, blockDimY, gridDimY);
  mlir::Value resX =
      mlir::arith::MulIOp::create(builder, loc, blockDimX, gridDimX);
  mlir::Value resZY = mlir::arith::MulIOp::create(builder, loc, resZ, resY);
  mlir::Value size = mlir::arith::MulIOp::create(builder, loc, resZY, resX);

  // tmp = ((blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x)) +
  //   blockIdx.x;
  // this_group.rank = tmp * ((blockDim.x * blockDim.y) * blockDim.z) +
  //   ((threadIdx.z * blockDim.y) * blockDim.x) +
  //   (threadIdx.y * blockDim.x) + threadIdx.x + 1;
  mlir::Value r1 =
      mlir::arith::MulIOp::create(builder, loc, blockIdZ, gridDimY);
  mlir::Value r2 = mlir::arith::MulIOp::create(builder, loc, r1, gridDimX);
  mlir::Value r3 =
      mlir::arith::MulIOp::create(builder, loc, blockIdY, gridDimX);
  mlir::Value r2r3 = mlir::arith::AddIOp::create(builder, loc, r2, r3);
  mlir::Value tmp = mlir::arith::AddIOp::create(builder, loc, r2r3, blockIdX);

  mlir::Value bXbY =
      mlir::arith::MulIOp::create(builder, loc, blockDimX, blockDimY);
  mlir::Value bXbYbZ =
      mlir::arith::MulIOp::create(builder, loc, bXbY, blockDimZ);
  mlir::Value tZbY =
      mlir::arith::MulIOp::create(builder, loc, threadIdZ, blockDimY);
  mlir::Value tZbYbX =
      mlir::arith::MulIOp::create(builder, loc, tZbY, blockDimX);
  mlir::Value tYbX =
      mlir::arith::MulIOp::create(builder, loc, threadIdY, blockDimX);
  mlir::Value rank = mlir::arith::MulIOp::create(builder, loc, tmp, bXbYbZ);
  rank = mlir::arith::AddIOp::create(builder, loc, rank, tZbYbX);
  rank = mlir::arith::AddIOp::create(builder, loc, rank, tYbX);
  rank = mlir::arith::AddIOp::create(builder, loc, rank, threadIdX);
  mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  rank = mlir::arith::AddIOp::create(builder, loc, rank, one);

  auto sizeFieldName = recTy.getTypeList()[1].first;
  mlir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  mlir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  mlir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  auto rankFieldName = recTy.getTypeList()[2].first;
  mlir::Type rankFieldTy = recTy.getTypeList()[2].second;
  mlir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THIS_THREAD_BLOCK
mlir::Value
CUDAIntrinsicLibrary::genThisThreadBlock(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 0);
  auto recTy = mlir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  mlir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  mlir::Type i32Ty = builder.getI32Type();

  // this_thread_block%size = blockDim.z * blockDim.y * blockDim.x;
  mlir::Value blockDimX = mlir::NVVM::BlockDimXOp::create(builder, loc, i32Ty);
  mlir::Value blockDimY = mlir::NVVM::BlockDimYOp::create(builder, loc, i32Ty);
  mlir::Value blockDimZ = mlir::NVVM::BlockDimZOp::create(builder, loc, i32Ty);
  mlir::Value size =
      mlir::arith::MulIOp::create(builder, loc, blockDimZ, blockDimY);
  size = mlir::arith::MulIOp::create(builder, loc, size, blockDimX);

  // this_thread_block%rank = ((threadIdx.z * blockDim.y) * blockDim.x) +
  //   (threadIdx.y * blockDim.x) + threadIdx.x + 1;
  mlir::Value threadIdX = mlir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  mlir::Value threadIdY = mlir::NVVM::ThreadIdYOp::create(builder, loc, i32Ty);
  mlir::Value threadIdZ = mlir::NVVM::ThreadIdZOp::create(builder, loc, i32Ty);
  mlir::Value r1 =
      mlir::arith::MulIOp::create(builder, loc, threadIdZ, blockDimY);
  mlir::Value r2 = mlir::arith::MulIOp::create(builder, loc, r1, blockDimX);
  mlir::Value r3 =
      mlir::arith::MulIOp::create(builder, loc, threadIdY, blockDimX);
  mlir::Value r2r3 = mlir::arith::AddIOp::create(builder, loc, r2, r3);
  mlir::Value rank = mlir::arith::AddIOp::create(builder, loc, r2r3, threadIdX);
  mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  rank = mlir::arith::AddIOp::create(builder, loc, rank, one);

  auto sizeFieldName = recTy.getTypeList()[1].first;
  mlir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  mlir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  mlir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  auto rankFieldName = recTy.getTypeList()[2].first;
  mlir::Type rankFieldTy = recTy.getTypeList()[2].second;
  mlir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THIS_WARP
mlir::Value
CUDAIntrinsicLibrary::genThisWarp(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 0);
  auto recTy = mlir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  mlir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  mlir::Type i32Ty = builder.getI32Type();

  // coalesced_group%size = 32
  mlir::Value size = builder.createIntegerConstant(loc, i32Ty, 32);
  auto sizeFieldName = recTy.getTypeList()[1].first;
  mlir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  mlir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  mlir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  // coalesced_group%rank = threadIdx.x & 31 + 1
  mlir::Value threadIdX = mlir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  mlir::Value mask = builder.createIntegerConstant(loc, i32Ty, 31);
  mlir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  mlir::Value masked =
      mlir::arith::AndIOp::create(builder, loc, threadIdX, mask);
  mlir::Value rank = mlir::arith::AddIOp::create(builder, loc, masked, one);
  auto rankFieldName = recTy.getTypeList()[2].first;
  mlir::Type rankFieldTy = recTy.getTypeList()[2].second;
  mlir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/mlir::ValueRange{});
  mlir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THREADFENCE, THREADFENCE_BLOCK, THREADFENCE_SYSTEM
template <mlir::NVVM::MemScopeKind scope>
void CUDAIntrinsicLibrary::genThreadFence(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  mlir::NVVM::MembarOp::create(builder, loc, scope);
}

// TMA_BULK_COMMIT_GROUP
void CUDAIntrinsicLibrary::genTMABulkCommitGroup(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  mlir::NVVM::CpAsyncBulkCommitGroupOp::create(builder, loc);
}

// TMA_BULK_G2S
void CUDAIntrinsicLibrary::genTMABulkG2S(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, fir::getBase(args[0]), mlir::NVVM::NVVMMemorySpace::Shared);
  mlir::Value dst =
      convertPtrToNVVMSpace(builder, loc, fir::getBase(args[2]),
                            mlir::NVVM::NVVMMemorySpace::SharedCluster);
  mlir::Value src = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[1]),
                                          mlir::NVVM::NVVMMemorySpace::Global);
  mlir::NVVM::CpAsyncBulkGlobalToSharedClusterOp::create(
      builder, loc, dst, src, barrier, fir::getBase(args[3]), {}, {});
}

static void genTMABulkLoad(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value barrier, mlir::Value src,
                           mlir::Value dst, mlir::Value nelem,
                           mlir::Value eleSize) {
  mlir::Value size = mlir::arith::MulIOp::create(builder, loc, nelem, eleSize);
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  barrier = builder.createConvert(loc, llvmPtrTy, barrier);
  dst = builder.createConvert(loc, llvmPtrTy, dst);
  src = builder.createConvert(loc, llvmPtrTy, src);
  mlir::NVVM::InlinePtxOp::create(
      builder, loc, mlir::TypeRange{}, {dst, src, size, barrier}, {},
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];",
      {});
  mlir::NVVM::InlinePtxOp::create(
      builder, loc, mlir::TypeRange{}, {barrier, size}, {},
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;", {});
}

// TMA_BULK_LOADC4
void CUDAIntrinsicLibrary::genTMABulkLoadC4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADC8
void CUDAIntrinsicLibrary::genTMABulkLoadC8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 16);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADI4
void CUDAIntrinsicLibrary::genTMABulkLoadI4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADI8
void CUDAIntrinsicLibrary::genTMABulkLoadI8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR2
void CUDAIntrinsicLibrary::genTMABulkLoadR2(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 2);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR4
void CUDAIntrinsicLibrary::genTMABulkLoadR4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR8
void CUDAIntrinsicLibrary::genTMABulkLoadR8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_S2G
void CUDAIntrinsicLibrary::genTMABulkS2G(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value src = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[0]),
                                          mlir::NVVM::NVVMMemorySpace::Shared);
  mlir::Value dst = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[1]),
                                          mlir::NVVM::NVVMMemorySpace::Global);
  mlir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(
      builder, loc, dst, src, fir::getBase(args[2]), {}, {});

  mlir::NVVM::InlinePtxOp::create(builder, loc, mlir::TypeRange{}, {}, {},
                                  "cp.async.bulk.commit_group;", {});
  mlir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc,
                                             builder.getI32IntegerAttr(0), {});
}

static void genTMABulkStore(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Value dst, mlir::Value count,
                            mlir::Value eleSize) {
  mlir::Value size = mlir::arith::MulIOp::create(builder, loc, eleSize, count);
  src = convertPtrToNVVMSpace(builder, loc, src,
                              mlir::NVVM::NVVMMemorySpace::Shared);
  dst = convertPtrToNVVMSpace(builder, loc, dst,
                              mlir::NVVM::NVVMMemorySpace::Global);
  mlir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(builder, loc, dst, src,
                                                     size, {}, {});
  mlir::NVVM::InlinePtxOp::create(builder, loc, mlir::TypeRange{}, {}, {},
                                  "cp.async.bulk.commit_group;", {});
  mlir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc,
                                             builder.getI32IntegerAttr(0), {});
}

// TMA_BULK_STORE_C4
void CUDAIntrinsicLibrary::genTMABulkStoreC4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_C8
void CUDAIntrinsicLibrary::genTMABulkStoreC8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 16);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_I4
void CUDAIntrinsicLibrary::genTMABulkStoreI4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_I8
void CUDAIntrinsicLibrary::genTMABulkStoreI8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R2
void CUDAIntrinsicLibrary::genTMABulkStoreR2(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 2);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R4
void CUDAIntrinsicLibrary::genTMABulkStoreR4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R8
void CUDAIntrinsicLibrary::genTMABulkStoreR8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_WAIT_GROUP
void CUDAIntrinsicLibrary::genTMABulkWaitGroup(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  auto group = builder.getIntegerAttr(builder.getI32Type(), 0);
  mlir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc, group, {});
}

// ALL_SYNC, ANY_SYNC, BALLOT_SYNC
template <mlir::NVVM::VoteSyncKind kind>
mlir::Value
CUDAIntrinsicLibrary::genVoteSync(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  mlir::Value arg1 =
      fir::ConvertOp::create(builder, loc, builder.getI1Type(), args[1]);
  mlir::Type resTy = kind == mlir::NVVM::VoteSyncKind::ballot
                         ? builder.getI32Type()
                         : builder.getI1Type();
  auto voteRes =
      mlir::NVVM::VoteSyncOp::create(builder, loc, resTy, args[0], arg1, kind)
          .getResult();
  return fir::ConvertOp::create(builder, loc, resultType, voteRes);
}

} // namespace fir
