//===-- CUDAIntrinsicCall.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of AIIR for PowerPC
// intrinsics. Extensive use of AIIR interfaces and AIIR's coding style
// (https://aiir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUDAIntrinsicCall.h"
#include "flang/Evaluate/common.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Runtime/entry-names.h"
#include "aiir/Dialect/Index/IR/IndexOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"

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

static constexpr unsigned kTMAAlignment = 16;

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
         &CI::genVoteSync<aiir::NVVM::VoteSyncKind::all>),
     {{{"mask", asValue}, {"pred", asValue}}},
     /*isElemental=*/false},
    {"any_sync",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genVoteSync<aiir::NVVM::VoteSyncKind::any>),
     {{{"mask", asValue}, {"pred", asValue}}},
     /*isElemental=*/false},
    {"atomicadd_r4x2",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector<2>),
     {{{"a", asAddr}, {"v", asAddr}}},
     false},
    {"atomicadd_r4x4",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genAtomicAddVector4x4),
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
         &CI::genVoteSync<aiir::NVVM::VoteSyncKind::ballot>),
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
         &CI::genNVVMTime<aiir::NVVM::ClockOp>),
     {},
     /*isElemental=*/false},
    {"clock64",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genNVVMTime<aiir::NVVM::Clock64Op>),
     {},
     /*isElemental=*/false},
    {"cluster_block_index",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genClusterBlockIndex),
     {},
     /*isElemental=*/false},
    {"cluster_dim_blocks",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genClusterDimBlocks),
     {},
     /*isElemental=*/false},
    {"cudagetstreamdefaultarg",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genCUDAGetDefaultStreamArg),
     {{{"devptr", asAddr}}},
     /*isElemental=*/false},
    {"cudagetstreamdefaultnull",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genCUDAGetDefaultStreamNull),
     {},
     /*isElemental=*/false},
    {"cudasetstreamarray",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genCUDASetDefaultStreamArray),
     {{{"devptr", asAddr}, {"stream", asValue}}},
     /*isElemental=*/false},
    {"cudasetstreamdefault",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genCUDASetDefaultStream),
     {{{"stream", asValue}}},
     /*isElemental=*/false},
    {"cudastreamdestroy",
     static_cast<CUDAIntrinsicLibrary::ExtendedGenerator>(
         &CI::genCUDAStreamDestroy),
     {{{"stream", asValue}}},
     /*isElemental=*/false},
    {"fence_proxy_async",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genFenceProxyAsync),
     {},
     /*isElemental=*/false},
    {"globaltimer",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(
         &CI::genNVVMTime<aiir::NVVM::GlobalTimerOp>),
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
    {"this_cluster",
     static_cast<CUDAIntrinsicLibrary::ElementalGenerator>(&CI::genThisCluster),
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
         &CI::genThreadFence<aiir::NVVM::MemScopeKind::GPU>),
     {},
     /*isElemental=*/false},
    {"threadfence_block",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genThreadFence<aiir::NVVM::MemScopeKind::CTA>),
     {},
     /*isElemental=*/false},
    {"threadfence_system",
     static_cast<CUDAIntrinsicLibrary::SubroutineGenerator>(
         &CI::genThreadFence<aiir::NVVM::MemScopeKind::SYS>),
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

static aiir::Value convertPtrToNVVMSpace(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         aiir::Value barrier,
                                         aiir::NVVM::NVVMMemorySpace space) {
  aiir::Value llvmPtr = fir::ConvertOp::create(
      builder, loc, aiir::LLVM::LLVMPointerType::get(builder.getContext()),
      barrier);
  aiir::Value addrCast = aiir::LLVM::AddrSpaceCastOp::create(
      builder, loc,
      aiir::LLVM::LLVMPointerType::get(builder.getContext(),
                                       static_cast<unsigned>(space)),
      llvmPtr);
  return addrCast;
}

static aiir::Value genAtomBinOp(fir::FirOpBuilder &builder, aiir::Location &loc,
                                aiir::LLVM::AtomicBinOp binOp, aiir::Value arg0,
                                aiir::Value arg1) {
  auto llvmPointerType = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  arg0 = builder.createConvert(loc, llvmPointerType, arg0);
  return aiir::LLVM::AtomicRMWOp::create(builder, loc, binOp, arg0, arg1,
                                         aiir::LLVM::AtomicOrdering::seq_cst);
}

// ATOMICADD
aiir::Value
CUDAIntrinsicLibrary::genAtomicAdd(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::LLVM::AtomicBinOp binOp =
      aiir::isa<aiir::IntegerType>(args[1].getType())
          ? aiir::LLVM::AtomicBinOp::add
          : aiir::LLVM::AtomicBinOp::fadd;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicAddR2(aiir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  aiir::Value a = fir::getBase(args[0]);

  if (aiir::isa<fir::BaseBoxType>(a.getType())) {
    a = fir::BoxAddrOp::create(builder, loc, a);
  }

  auto loc = builder.getUnknownLoc();
  auto f16Ty = builder.getF16Type();
  auto i32Ty = builder.getI32Type();
  auto vecF16Ty = aiir::VectorType::get({2}, f16Ty);
  aiir::Type idxTy = builder.getIndexType();
  auto f16RefTy = fir::ReferenceType::get(f16Ty);
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto one = builder.createIntegerConstant(loc, idxTy, 1);
  auto v1Coord = fir::CoordinateOp::create(builder, loc, f16RefTy,
                                           fir::getBase(args[1]), zero);
  auto v2Coord = fir::CoordinateOp::create(builder, loc, f16RefTy,
                                           fir::getBase(args[1]), one);
  auto v1 = fir::LoadOp::create(builder, loc, v1Coord);
  auto v2 = fir::LoadOp::create(builder, loc, v2Coord);
  aiir::Value undef = aiir::LLVM::UndefOp::create(builder, loc, vecF16Ty);
  aiir::Value vec1 = aiir::LLVM::InsertElementOp::create(
      builder, loc, undef, v1, builder.createIntegerConstant(loc, i32Ty, 0));
  aiir::Value vec2 = aiir::LLVM::InsertElementOp::create(
      builder, loc, vec1, v2, builder.createIntegerConstant(loc, i32Ty, 1));
  auto res = genAtomBinOp(builder, loc, aiir::LLVM::AtomicBinOp::fadd, a, vec2);
  auto i32VecTy = aiir::VectorType::get({1}, i32Ty);
  aiir::Value vecI32 =
      aiir::vector::BitCastOp::create(builder, loc, i32VecTy, res);
  return aiir::vector::ExtractOp::create(builder, loc, vecI32,
                                         aiir::ArrayRef<int64_t>{0});
}

// ATOMICADDVECTOR
template <int extent>
fir::ExtendedValue CUDAIntrinsicLibrary::genAtomicAddVector(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value res = fir::AllocaOp::create(
      builder, loc, fir::SequenceType::get({extent}, resultType));
  aiir::Value a = fir::getBase(args[0]);
  if (aiir::isa<fir::BaseBoxType>(a.getType())) {
    a = fir::BoxAddrOp::create(builder, loc, a);
  }
  auto vecTy = aiir::VectorType::get({extent}, resultType);
  auto refTy = fir::ReferenceType::get(resultType);
  aiir::Type i32Ty = builder.getI32Type();
  aiir::Type idxTy = builder.getIndexType();

  // Extract the values from the array.
  llvm::SmallVector<aiir::Value> values;
  for (unsigned i = 0; i < extent; ++i) {
    aiir::Value pos = builder.createIntegerConstant(loc, idxTy, i);
    aiir::Value coord = fir::CoordinateOp::create(builder, loc, refTy,
                                                  fir::getBase(args[1]), pos);
    aiir::Value value = fir::LoadOp::create(builder, loc, coord);
    values.push_back(value);
  }
  // Pack extracted values into a vector to call the atomic add.
  aiir::Value undef = aiir::LLVM::UndefOp::create(builder, loc, vecTy);
  for (unsigned i = 0; i < extent; ++i) {
    aiir::Value insert = aiir::LLVM::InsertElementOp::create(
        builder, loc, undef, values[i],
        builder.createIntegerConstant(loc, i32Ty, i));
    undef = insert;
  }
  // Atomic operation with a vector of values.
  aiir::Value add =
      genAtomBinOp(builder, loc, aiir::LLVM::AtomicBinOp::fadd, a, undef);
  // Store results in the result array.
  for (unsigned i = 0; i < extent; ++i) {
    aiir::Value r = aiir::LLVM::ExtractElementOp::create(
        builder, loc, add, builder.createIntegerConstant(loc, i32Ty, i));
    aiir::Value c = fir::CoordinateOp::create(
        builder, loc, refTy, res, builder.createIntegerConstant(loc, idxTy, i));
    fir::StoreOp::create(builder, loc, r, c);
  }
  aiir::Value ext = builder.createIntegerConstant(loc, idxTy, extent);
  return fir::ArrayBoxValue(res, {ext});
}

// ATOMICADDVECTOR4x4
fir::ExtendedValue CUDAIntrinsicLibrary::genAtomicAddVector4x4(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value a = fir::getBase(args[0]);
  if (aiir::isa<fir::BaseBoxType>(a.getType()))
    a = fir::BoxAddrOp::create(builder, loc, a);

  const unsigned extent = 4;
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  aiir::Value ptr = builder.createConvert(loc, llvmPtrTy, a);
  aiir::Type f32Ty = builder.getF32Type();
  aiir::Type idxTy = builder.getIndexType();
  aiir::Type refTy = fir::ReferenceType::get(f32Ty);
  llvm::SmallVector<aiir::Value> values;
  for (unsigned i = 0; i < extent; ++i) {
    aiir::Value pos = builder.createIntegerConstant(loc, idxTy, i);
    aiir::Value coord = fir::CoordinateOp::create(builder, loc, refTy,
                                                  fir::getBase(args[1]), pos);
    aiir::Value value = fir::LoadOp::create(builder, loc, coord);
    values.push_back(value);
  }

  auto inlinePtx = aiir::NVVM::InlinePtxOp::create(
      builder, loc, {f32Ty, f32Ty, f32Ty, f32Ty},
      {ptr, values[0], values[1], values[2], values[3]}, {},
      "atom.add.v4.f32 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8};", {});

  llvm::SmallVector<aiir::Value> results;
  results.push_back(inlinePtx.getResult(0));
  results.push_back(inlinePtx.getResult(1));
  results.push_back(inlinePtx.getResult(2));
  results.push_back(inlinePtx.getResult(3));

  aiir::Type vecF32Ty = aiir::VectorType::get({extent}, f32Ty);
  aiir::Value undef = aiir::LLVM::UndefOp::create(builder, loc, vecF32Ty);
  aiir::Type i32Ty = builder.getI32Type();
  for (unsigned i = 0; i < extent; ++i)
    undef = aiir::LLVM::InsertElementOp::create(
        builder, loc, undef, results[i],
        builder.createIntegerConstant(loc, i32Ty, i));

  auto i128Ty = builder.getIntegerType(128);
  auto i128VecTy = aiir::VectorType::get({1}, i128Ty);
  aiir::Value vec128 =
      aiir::vector::BitCastOp::create(builder, loc, i128VecTy, undef);
  return aiir::vector::ExtractOp::create(builder, loc, vec128,
                                         aiir::ArrayRef<int64_t>{0});
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicAnd(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  assert(aiir::isa<aiir::IntegerType>(args[1].getType()));

  aiir::LLVM::AtomicBinOp binOp = aiir::LLVM::AtomicBinOp::_and;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicOr(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  assert(aiir::isa<aiir::IntegerType>(args[1].getType()));

  aiir::LLVM::AtomicBinOp binOp = aiir::LLVM::AtomicBinOp::_or;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICCAS
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicCas(aiir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto successOrdering = aiir::LLVM::AtomicOrdering::acq_rel;
  auto failureOrdering = aiir::LLVM::AtomicOrdering::monotonic;
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(resultType.getContext());

  aiir::Value arg0 = fir::getBase(args[0]);
  aiir::Value arg1 = fir::getBase(args[1]);
  aiir::Value arg2 = fir::getBase(args[2]);

  auto bitCastFloat = [&](aiir::Value arg) -> aiir::Value {
    if (aiir::isa<aiir::Float32Type>(arg.getType()))
      return aiir::LLVM::BitcastOp::create(builder, loc, builder.getI32Type(),
                                           arg);
    if (aiir::isa<aiir::Float64Type>(arg.getType()))
      return aiir::LLVM::BitcastOp::create(builder, loc, builder.getI64Type(),
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
      aiir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy, arg0)
          .getResult(0);
  auto cmpxchg = aiir::LLVM::AtomicCmpXchgOp::create(
      builder, loc, address, arg1, arg2, successOrdering, failureOrdering);
  aiir::Value boolResult =
      aiir::LLVM::ExtractValueOp::create(builder, loc, cmpxchg, 1);
  return builder.createConvert(loc, resultType, boolResult);
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicDec(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  assert(aiir::isa<aiir::IntegerType>(args[1].getType()));

  aiir::LLVM::AtomicBinOp binOp = aiir::LLVM::AtomicBinOp::udec_wrap;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICEXCH
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicExch(aiir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value arg0 = fir::getBase(args[0]);
  aiir::Value arg1 = fir::getBase(args[1]);
  assert(arg1.getType().isIntOrFloat());

  aiir::LLVM::AtomicBinOp binOp = aiir::LLVM::AtomicBinOp::xchg;
  return genAtomBinOp(builder, loc, binOp, arg0, arg1);
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicInc(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  assert(aiir::isa<aiir::IntegerType>(args[1].getType()));

  aiir::LLVM::AtomicBinOp binOp = aiir::LLVM::AtomicBinOp::uinc_wrap;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicMax(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  aiir::LLVM::AtomicBinOp binOp =
      aiir::isa<aiir::IntegerType>(args[1].getType())
          ? aiir::LLVM::AtomicBinOp::max
          : aiir::LLVM::AtomicBinOp::fmax;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

aiir::Value
CUDAIntrinsicLibrary::genAtomicMin(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  aiir::LLVM::AtomicBinOp binOp =
      aiir::isa<aiir::IntegerType>(args[1].getType())
          ? aiir::LLVM::AtomicBinOp::min
          : aiir::LLVM::AtomicBinOp::fmin;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICSUB
aiir::Value
CUDAIntrinsicLibrary::genAtomicSub(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::LLVM::AtomicBinOp binOp =
      aiir::isa<aiir::IntegerType>(args[1].getType())
          ? aiir::LLVM::AtomicBinOp::sub
          : aiir::LLVM::AtomicBinOp::fsub;
  return genAtomBinOp(builder, loc, binOp, args[0], args[1]);
}

// ATOMICXOR
fir::ExtendedValue
CUDAIntrinsicLibrary::genAtomicXor(aiir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value arg0 = fir::getBase(args[0]);
  aiir::Value arg1 = fir::getBase(args[1]);
  return genAtomBinOp(builder, loc, aiir::LLVM::AtomicBinOp::_xor, arg0, arg1);
}

// BARRIER_ARRIVE
aiir::Value
CUDAIntrinsicLibrary::genBarrierArrive(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, args[0], aiir::NVVM::NVVMMemorySpace::Shared);
  return aiir::NVVM::MBarrierArriveOp::create(builder, loc, resultType, barrier)
      .getResult(0);
}

// BARRIER_ARRIBVE_CNT
aiir::Value
CUDAIntrinsicLibrary::genBarrierArriveCnt(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, args[0], aiir::NVVM::NVVMMemorySpace::Shared);
  return aiir::NVVM::InlinePtxOp::create(builder, loc, {resultType},
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
  aiir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, fir::getBase(args[0]), aiir::NVVM::NVVMMemorySpace::Shared);
  aiir::NVVM::MBarrierInitOp::create(builder, loc, barrier,
                                     fir::getBase(args[1]), {});
  auto kind = aiir::NVVM::ProxyKindAttr::get(
      builder.getContext(), aiir::NVVM::ProxyKind::async_shared);
  auto space = aiir::NVVM::SharedSpaceAttr::get(
      builder.getContext(), aiir::NVVM::SharedSpace::shared_cta);
  aiir::NVVM::FenceProxyOp::create(builder, loc, kind, space);
}

// BARRIER_TRY_WAIT
aiir::Value
CUDAIntrinsicLibrary::genBarrierTryWait(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  fir::StoreOp::create(builder, loc, zero, res);
  aiir::Value ns =
      builder.createIntegerConstant(loc, builder.getI32Type(), 1000000);
  aiir::Value load = fir::LoadOp::create(builder, loc, res);
  auto whileOp = aiir::scf::WhileOp::create(
      builder, loc, aiir::TypeRange{resultType}, aiir::ValueRange{load});
  aiir::Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
  aiir::Value beforeArg = beforeBlock->addArgument(resultType, loc);
  builder.setInsertionPointToStart(beforeBlock);
  aiir::Value condition = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, beforeArg, zero);
  aiir::scf::ConditionOp::create(builder, loc, condition, beforeArg);
  aiir::Block *afterBlock = builder.createBlock(&whileOp.getAfter());
  afterBlock->addArgument(resultType, loc);
  builder.setInsertionPointToStart(afterBlock);
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  auto barrier = builder.createConvert(loc, llvmPtrTy, args[0]);
  aiir::Value ret = aiir::NVVM::InlinePtxOp::create(
                        builder, loc, {resultType}, {barrier, args[1], ns}, {},
                        "{\n"
                        "  .reg .pred p;\n"
                        "  mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n"
                        "  selp.b32 %0, 1, 0, p;\n"
                        "}",
                        {})
                        .getResult(0);
  aiir::scf::YieldOp::create(builder, loc, ret);
  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(0);
}

// BARRIER_TRY_WAIT_SLEEP
aiir::Value
CUDAIntrinsicLibrary::genBarrierTryWaitSleep(aiir::Type resultType,
                                             llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  auto barrier = builder.createConvert(loc, llvmPtrTy, args[0]);
  return aiir::NVVM::InlinePtxOp::create(
             builder, loc, {resultType}, {barrier, args[1], args[2]}, {},
             "{\n"
             "  .reg .pred p;\n"
             "  mbarrier.try_wait.shared.b64 p, [%1], %2, %3;\n"
             "  selp.b32 %0, 1, 0, p;\n"
             "}",
             {})
      .getResult(0);
}

static void insertValueAtPos(fir::FirOpBuilder &builder, aiir::Location loc,
                             fir::RecordType recTy, aiir::Value base,
                             aiir::Value dim, unsigned fieldPos) {
  auto fieldName = recTy.getTypeList()[fieldPos].first;
  aiir::Type fieldTy = recTy.getTypeList()[fieldPos].second;
  aiir::Type fieldIndexType = fir::FieldType::get(base.getContext());
  aiir::Value fieldIndex =
      fir::FieldIndexOp::create(builder, loc, fieldIndexType, fieldName, recTy,
                                /*typeParams=*/aiir::ValueRange{});
  aiir::Value coord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(fieldTy), base, fieldIndex);
  fir::StoreOp::create(builder, loc, dim, coord);
}

// CLUSTER_BLOCK_INDEX
aiir::Value
CUDAIntrinsicLibrary::genClusterBlockIndex(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();
  aiir::Value x = aiir::NVVM::BlockInClusterIdXOp::create(builder, loc, i32Ty);
  aiir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  x = aiir::arith::AddIOp::create(builder, loc, x, one);
  insertValueAtPos(builder, loc, recTy, res, x, 0);
  aiir::Value y = aiir::NVVM::BlockInClusterIdYOp::create(builder, loc, i32Ty);
  y = aiir::arith::AddIOp::create(builder, loc, y, one);
  insertValueAtPos(builder, loc, recTy, res, y, 1);
  aiir::Value z = aiir::NVVM::BlockInClusterIdZOp::create(builder, loc, i32Ty);
  z = aiir::arith::AddIOp::create(builder, loc, z, one);
  insertValueAtPos(builder, loc, recTy, res, z, 2);
  return res;
}

// CLUSTER_DIM_BLOCKS
aiir::Value
CUDAIntrinsicLibrary::genClusterDimBlocks(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();
  aiir::Value x = aiir::NVVM::ClusterDimBlocksXOp::create(builder, loc, i32Ty);
  insertValueAtPos(builder, loc, recTy, res, x, 0);
  aiir::Value y = aiir::NVVM::ClusterDimBlocksYOp::create(builder, loc, i32Ty);
  insertValueAtPos(builder, loc, recTy, res, y, 1);
  aiir::Value z = aiir::NVVM::ClusterDimBlocksZOp::create(builder, loc, i32Ty);
  insertValueAtPos(builder, loc, recTy, res, z, 2);
  return res;
}

// CUDASETSTREAMDEFAULT
fir::ExtendedValue CUDAIntrinsicLibrary::genCUDASetDefaultStream(
    aiir::Type resTy, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value stream = fir::getBase(args[0]);
  aiir::Type i64Ty = builder.getI64Type();
  auto ctx = builder.getContext();
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {i64Ty}, {resTy});
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFSetDefaultStream), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {stream});
  return call.getResult(0);
}

// CUDASETSTREAMARRAY
fir::ExtendedValue CUDAIntrinsicLibrary::genCUDASetDefaultStreamArray(
    aiir::Type resTy, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value arg = fir::getBase(args[0]);
  aiir::Value stream = fir::getBase(args[1]);

  if (aiir::isa<fir::BaseBoxType>(arg.getType()))
    arg = fir::BoxAddrOp::create(builder, loc, arg);
  aiir::Type i64Ty = builder.getI64Type();
  aiir::Type i32Ty = builder.getI32Type();
  auto ctx = builder.getContext();
  aiir::Type voidPtrTy =
      fir::LLVMPointerType::get(ctx, aiir::IntegerType::get(ctx, 8));
  aiir::FunctionType ftype =
      aiir::FunctionType::get(ctx, {voidPtrTy, i64Ty}, {i32Ty});
  aiir::Value voidPtr = builder.createConvert(loc, voidPtrTy, arg);
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFSetAssociatedStream), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {voidPtr, stream});
  return call.getResult(0);
}

// CUDASTREAMDESTROY
fir::ExtendedValue CUDAIntrinsicLibrary::genCUDAStreamDestroy(
    aiir::Type resTy, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value stream = fir::getBase(args[0]);
  aiir::Type i64Ty = builder.getI64Type();
  auto ctx = builder.getContext();
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {i64Ty}, {resTy});
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFStreamDestroy), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {stream});
  return call.getResult(0);
}

// CUDASTREAMSYNCHRONIZE
fir::ExtendedValue CUDAIntrinsicLibrary::genCUDAStreamSynchronize(
    aiir::Type resTy, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value stream = fir::getBase(args[0]);
  aiir::Type i64Ty = builder.getI64Type();
  auto ctx = builder.getContext();
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {i64Ty}, {resTy});
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFStreamSynchronize), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {stream});
  return call.getResult(0);
}

// CUDASTREAMSYNCHRONIZENULL
aiir::Value CUDAIntrinsicLibrary::genCUDAStreamSynchronizeNull(
    aiir::Type resTy, llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto ctx = builder.getContext();
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {}, {resTy});
  auto funcOp = builder.createFunction(
      loc, RTNAME_STRING(CUFStreamSynchronizeNull), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {});
  return call.getResult(0);
}

// CUDAGETDEFAULTSTREAMARG
fir::ExtendedValue CUDAIntrinsicLibrary::genCUDAGetDefaultStreamArg(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value devptr = fir::getBase(args[0]);
  aiir::Type i64Ty = builder.getI64Type();
  auto ctx = builder.getContext();
  aiir::Type voidPtrTy =
      fir::LLVMPointerType::get(ctx, aiir::IntegerType::get(ctx, 8));
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {voidPtrTy}, {i64Ty});
  aiir::Value voidPtr = builder.createConvert(loc, voidPtrTy, devptr);
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFGetAssociatedStream), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {voidPtr});
  return call.getResult(0);
}

// CUDAGETDEFAULTSTREAMNULL
aiir::Value CUDAIntrinsicLibrary::genCUDAGetDefaultStreamNull(
    aiir::Type resultType, llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  aiir::Type i64Ty = builder.getI64Type();
  auto ctx = builder.getContext();
  aiir::FunctionType ftype = aiir::FunctionType::get(ctx, {}, {i64Ty});
  auto funcOp =
      builder.createFunction(loc, RTNAME_STRING(CUFGetDefaultStream), ftype);
  auto call = fir::CallOp::create(builder, loc, funcOp, {});
  return call.getResult(0);
}

// FENCE_PROXY_ASYNC
void CUDAIntrinsicLibrary::genFenceProxyAsync(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  auto kind = aiir::NVVM::ProxyKindAttr::get(
      builder.getContext(), aiir::NVVM::ProxyKind::async_shared);
  auto space = aiir::NVVM::SharedSpaceAttr::get(
      builder.getContext(), aiir::NVVM::SharedSpace::shared_cta);
  aiir::NVVM::FenceProxyOp::create(builder, loc, kind, space);
}

// __LDCA, __LDCS, __LDLU, __LDCV
template <const char *fctName, int extent>
fir::ExtendedValue
CUDAIntrinsicLibrary::genLDXXFunc(aiir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Type resTy = fir::SequenceType::get(extent, resultType);
  aiir::Value arg = fir::getBase(args[0]);
  aiir::Value res = fir::AllocaOp::create(builder, loc, resTy);
  if (aiir::isa<fir::BaseBoxType>(arg.getType()))
    arg = fir::BoxAddrOp::create(builder, loc, arg);
  aiir::Type refResTy = fir::ReferenceType::get(resTy);
  aiir::FunctionType ftype =
      aiir::FunctionType::get(arg.getContext(), {refResTy, refResTy}, {});
  auto funcOp = builder.createFunction(loc, fctName, ftype);
  llvm::SmallVector<aiir::Value> funcArgs;
  funcArgs.push_back(res);
  funcArgs.push_back(arg);
  fir::CallOp::create(builder, loc, funcOp, funcArgs);
  aiir::Value ext =
      builder.createIntegerConstant(loc, builder.getIndexType(), extent);
  return fir::ArrayBoxValue(res, {ext});
}

// CLOCK, CLOCK64, GLOBALTIMER
template <typename OpTy>
aiir::Value
CUDAIntrinsicLibrary::genNVVMTime(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0 && "expect no arguments");
  return OpTy::create(builder, loc, resultType).getResult();
}

// MATCH_ALL_SYNC
aiir::Value
CUDAIntrinsicLibrary::genMatchAllSync(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);
  bool is32 = args[1].getType().isInteger(32) || args[1].getType().isF32();

  aiir::Type i1Ty = builder.getI1Type();
  aiir::AIIRContext *context = builder.getContext();

  aiir::Value arg1 = args[1];
  if (arg1.getType().isF32() || arg1.getType().isF64())
    arg1 = fir::ConvertOp::create(
        builder, loc, is32 ? builder.getI32Type() : builder.getI64Type(), arg1);

  aiir::Type retTy =
      aiir::LLVM::LLVMStructType::getLiteral(context, {resultType, i1Ty});
  auto match =
      aiir::NVVM::MatchSyncOp::create(builder, loc, retTy, args[0], arg1,
                                      aiir::NVVM::MatchSyncKind::all)
          .getResult();
  auto value = aiir::LLVM::ExtractValueOp::create(builder, loc, match, 0);
  auto pred = aiir::LLVM::ExtractValueOp::create(builder, loc, match, 1);
  auto conv = aiir::LLVM::ZExtOp::create(builder, loc, resultType, pred);
  fir::StoreOp::create(builder, loc, conv, args[2]);
  return value;
}

// MATCH_ANY_SYNC
aiir::Value
CUDAIntrinsicLibrary::genMatchAnySync(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  bool is32 = args[1].getType().isInteger(32) || args[1].getType().isF32();

  aiir::Value arg1 = args[1];
  if (arg1.getType().isF32() || arg1.getType().isF64())
    arg1 = fir::ConvertOp::create(
        builder, loc, is32 ? builder.getI32Type() : builder.getI64Type(), arg1);

  return aiir::NVVM::MatchSyncOp::create(builder, loc, resultType, args[0],
                                         arg1, aiir::NVVM::MatchSyncKind::any)
      .getResult();
}

// SYNCTHREADS
void CUDAIntrinsicLibrary::genSyncThreads(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  aiir::NVVM::Barrier0Op::create(builder, loc);
}

// SYNCTHREADS_AND
aiir::Value
CUDAIntrinsicLibrary::genSyncThreadsAnd(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  aiir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return aiir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             aiir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), aiir::NVVM::BarrierReduction::AND),
             arg)
      .getResult(0);
}

// SYNCTHREADS_COUNT
aiir::Value
CUDAIntrinsicLibrary::genSyncThreadsCount(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  aiir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return aiir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             aiir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), aiir::NVVM::BarrierReduction::POPC),
             arg)
      .getResult(0);
}

// SYNCTHREADS_OR
aiir::Value
CUDAIntrinsicLibrary::genSyncThreadsOr(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  aiir::Value arg = builder.createConvert(loc, builder.getI32Type(), args[0]);
  return aiir::NVVM::BarrierOp::create(
             builder, loc, resultType, {}, {},
             aiir::NVVM::BarrierReductionAttr::get(
                 builder.getContext(), aiir::NVVM::BarrierReduction::OR),
             arg)
      .getResult(0);
}

// SYNCWARP
void CUDAIntrinsicLibrary::genSyncWarp(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::NVVM::SyncWarpOp::create(builder, loc, fir::getBase(args[0]));
}

// THIS_CLUSTER
aiir::Value
CUDAIntrinsicLibrary::genThisCluster(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();

  // SIZE
  aiir::Value size = aiir::NVVM::ClusterDim::create(builder, loc, i32Ty);
  auto sizeFieldName = recTy.getTypeList()[1].first;
  aiir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  aiir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  aiir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  // RANK
  aiir::Value rank = aiir::NVVM::ClusterId::create(builder, loc, i32Ty);
  aiir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, one);
  auto rankFieldName = recTy.getTypeList()[2].first;
  aiir::Type rankFieldTy = recTy.getTypeList()[2].second;
  aiir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);

  return res;
}

// THIS_GRID
aiir::Value
CUDAIntrinsicLibrary::genThisGrid(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();

  aiir::Value threadIdX = aiir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  aiir::Value threadIdY = aiir::NVVM::ThreadIdYOp::create(builder, loc, i32Ty);
  aiir::Value threadIdZ = aiir::NVVM::ThreadIdZOp::create(builder, loc, i32Ty);

  aiir::Value blockIdX = aiir::NVVM::BlockIdXOp::create(builder, loc, i32Ty);
  aiir::Value blockIdY = aiir::NVVM::BlockIdYOp::create(builder, loc, i32Ty);
  aiir::Value blockIdZ = aiir::NVVM::BlockIdZOp::create(builder, loc, i32Ty);

  aiir::Value blockDimX = aiir::NVVM::BlockDimXOp::create(builder, loc, i32Ty);
  aiir::Value blockDimY = aiir::NVVM::BlockDimYOp::create(builder, loc, i32Ty);
  aiir::Value blockDimZ = aiir::NVVM::BlockDimZOp::create(builder, loc, i32Ty);
  aiir::Value gridDimX = aiir::NVVM::GridDimXOp::create(builder, loc, i32Ty);
  aiir::Value gridDimY = aiir::NVVM::GridDimYOp::create(builder, loc, i32Ty);
  aiir::Value gridDimZ = aiir::NVVM::GridDimZOp::create(builder, loc, i32Ty);

  // this_grid.size = ((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y)) *
  // (blockDim.x * gridDim.x);
  aiir::Value resZ =
      aiir::arith::MulIOp::create(builder, loc, blockDimZ, gridDimZ);
  aiir::Value resY =
      aiir::arith::MulIOp::create(builder, loc, blockDimY, gridDimY);
  aiir::Value resX =
      aiir::arith::MulIOp::create(builder, loc, blockDimX, gridDimX);
  aiir::Value resZY = aiir::arith::MulIOp::create(builder, loc, resZ, resY);
  aiir::Value size = aiir::arith::MulIOp::create(builder, loc, resZY, resX);

  // tmp = ((blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x)) +
  //   blockIdx.x;
  // this_group.rank = tmp * ((blockDim.x * blockDim.y) * blockDim.z) +
  //   ((threadIdx.z * blockDim.y) * blockDim.x) +
  //   (threadIdx.y * blockDim.x) + threadIdx.x + 1;
  aiir::Value r1 =
      aiir::arith::MulIOp::create(builder, loc, blockIdZ, gridDimY);
  aiir::Value r2 = aiir::arith::MulIOp::create(builder, loc, r1, gridDimX);
  aiir::Value r3 =
      aiir::arith::MulIOp::create(builder, loc, blockIdY, gridDimX);
  aiir::Value r2r3 = aiir::arith::AddIOp::create(builder, loc, r2, r3);
  aiir::Value tmp = aiir::arith::AddIOp::create(builder, loc, r2r3, blockIdX);

  aiir::Value bXbY =
      aiir::arith::MulIOp::create(builder, loc, blockDimX, blockDimY);
  aiir::Value bXbYbZ =
      aiir::arith::MulIOp::create(builder, loc, bXbY, blockDimZ);
  aiir::Value tZbY =
      aiir::arith::MulIOp::create(builder, loc, threadIdZ, blockDimY);
  aiir::Value tZbYbX =
      aiir::arith::MulIOp::create(builder, loc, tZbY, blockDimX);
  aiir::Value tYbX =
      aiir::arith::MulIOp::create(builder, loc, threadIdY, blockDimX);
  aiir::Value rank = aiir::arith::MulIOp::create(builder, loc, tmp, bXbYbZ);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, tZbYbX);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, tYbX);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, threadIdX);
  aiir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, one);

  auto sizeFieldName = recTy.getTypeList()[1].first;
  aiir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  aiir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  aiir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  auto rankFieldName = recTy.getTypeList()[2].first;
  aiir::Type rankFieldTy = recTy.getTypeList()[2].second;
  aiir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THIS_THREAD_BLOCK
aiir::Value
CUDAIntrinsicLibrary::genThisThreadBlock(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();

  // this_thread_block%size = blockDim.z * blockDim.y * blockDim.x;
  aiir::Value blockDimX = aiir::NVVM::BlockDimXOp::create(builder, loc, i32Ty);
  aiir::Value blockDimY = aiir::NVVM::BlockDimYOp::create(builder, loc, i32Ty);
  aiir::Value blockDimZ = aiir::NVVM::BlockDimZOp::create(builder, loc, i32Ty);
  aiir::Value size =
      aiir::arith::MulIOp::create(builder, loc, blockDimZ, blockDimY);
  size = aiir::arith::MulIOp::create(builder, loc, size, blockDimX);

  // this_thread_block%rank = ((threadIdx.z * blockDim.y) * blockDim.x) +
  //   (threadIdx.y * blockDim.x) + threadIdx.x + 1;
  aiir::Value threadIdX = aiir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  aiir::Value threadIdY = aiir::NVVM::ThreadIdYOp::create(builder, loc, i32Ty);
  aiir::Value threadIdZ = aiir::NVVM::ThreadIdZOp::create(builder, loc, i32Ty);
  aiir::Value r1 =
      aiir::arith::MulIOp::create(builder, loc, threadIdZ, blockDimY);
  aiir::Value r2 = aiir::arith::MulIOp::create(builder, loc, r1, blockDimX);
  aiir::Value r3 =
      aiir::arith::MulIOp::create(builder, loc, threadIdY, blockDimX);
  aiir::Value r2r3 = aiir::arith::AddIOp::create(builder, loc, r2, r3);
  aiir::Value rank = aiir::arith::AddIOp::create(builder, loc, r2r3, threadIdX);
  aiir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  rank = aiir::arith::AddIOp::create(builder, loc, rank, one);

  auto sizeFieldName = recTy.getTypeList()[1].first;
  aiir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  aiir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  aiir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  auto rankFieldName = recTy.getTypeList()[2].first;
  aiir::Type rankFieldTy = recTy.getTypeList()[2].second;
  aiir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THIS_WARP
aiir::Value
CUDAIntrinsicLibrary::genThisWarp(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  auto recTy = aiir::cast<fir::RecordType>(resultType);
  assert(recTy && "RecordType expepected");
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Type i32Ty = builder.getI32Type();

  // coalesced_group%size = 32
  aiir::Value size = builder.createIntegerConstant(loc, i32Ty, 32);
  auto sizeFieldName = recTy.getTypeList()[1].first;
  aiir::Type sizeFieldTy = recTy.getTypeList()[1].second;
  aiir::Type fieldIndexType = fir::FieldType::get(resultType.getContext());
  aiir::Value sizeFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, sizeFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value sizeCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(sizeFieldTy), res, sizeFieldIndex);
  fir::StoreOp::create(builder, loc, size, sizeCoord);

  // coalesced_group%rank = threadIdx.x & 31 + 1
  aiir::Value threadIdX = aiir::NVVM::ThreadIdXOp::create(builder, loc, i32Ty);
  aiir::Value mask = builder.createIntegerConstant(loc, i32Ty, 31);
  aiir::Value one = builder.createIntegerConstant(loc, i32Ty, 1);
  aiir::Value masked =
      aiir::arith::AndIOp::create(builder, loc, threadIdX, mask);
  aiir::Value rank = aiir::arith::AddIOp::create(builder, loc, masked, one);
  auto rankFieldName = recTy.getTypeList()[2].first;
  aiir::Type rankFieldTy = recTy.getTypeList()[2].second;
  aiir::Value rankFieldIndex = fir::FieldIndexOp::create(
      builder, loc, fieldIndexType, rankFieldName, recTy,
      /*typeParams=*/aiir::ValueRange{});
  aiir::Value rankCoord = fir::CoordinateOp::create(
      builder, loc, builder.getRefType(rankFieldTy), res, rankFieldIndex);
  fir::StoreOp::create(builder, loc, rank, rankCoord);
  return res;
}

// THREADFENCE, THREADFENCE_BLOCK, THREADFENCE_SYSTEM
template <aiir::NVVM::MemScopeKind scope>
void CUDAIntrinsicLibrary::genThreadFence(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  aiir::NVVM::MembarOp::create(builder, loc, scope);
}

// TMA_BULK_COMMIT_GROUP
void CUDAIntrinsicLibrary::genTMABulkCommitGroup(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  aiir::NVVM::CpAsyncBulkCommitGroupOp::create(builder, loc);
}

// TMA_BULK_G2S
void CUDAIntrinsicLibrary::genTMABulkG2S(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value barrier = convertPtrToNVVMSpace(
      builder, loc, fir::getBase(args[0]), aiir::NVVM::NVVMMemorySpace::Shared);
  aiir::Value dst =
      convertPtrToNVVMSpace(builder, loc, fir::getBase(args[2]),
                            aiir::NVVM::NVVMMemorySpace::SharedCluster);
  aiir::Value src = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[1]),
                                          aiir::NVVM::NVVMMemorySpace::Global);
  aiir::NVVM::CpAsyncBulkGlobalToSharedClusterOp::create(
      builder, loc, dst, src, barrier, fir::getBase(args[3]), {}, {});
}

static void setAlignment(aiir::Value ptr, unsigned alignment) {
  if (auto declareOp = aiir::dyn_cast<hlfir::DeclareOp>(ptr.getDefiningOp()))
    if (auto sharedOp = aiir::dyn_cast<cuf::SharedMemoryOp>(
            declareOp.getMemref().getDefiningOp()))
      sharedOp.setAlignment(alignment);
}

static void genTMABulkLoad(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value barrier, aiir::Value src,
                           aiir::Value dst, aiir::Value nelem,
                           aiir::Value eleSize) {
  aiir::Value size = aiir::arith::MulIOp::create(builder, loc, nelem, eleSize);
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  barrier = builder.createConvert(loc, llvmPtrTy, barrier);
  setAlignment(dst, kTMAAlignment);
  dst = builder.createConvert(loc, llvmPtrTy, dst);
  src = builder.createConvert(loc, llvmPtrTy, src);
  aiir::NVVM::InlinePtxOp::create(
      builder, loc, aiir::TypeRange{}, {dst, src, size, barrier}, {},
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];",
      {});
  aiir::NVVM::InlinePtxOp::create(
      builder, loc, aiir::TypeRange{}, {barrier, size}, {},
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;", {});
}

// TMA_BULK_LOADC4
void CUDAIntrinsicLibrary::genTMABulkLoadC4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADC8
void CUDAIntrinsicLibrary::genTMABulkLoadC8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 16);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADI4
void CUDAIntrinsicLibrary::genTMABulkLoadI4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADI8
void CUDAIntrinsicLibrary::genTMABulkLoadI8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR2
void CUDAIntrinsicLibrary::genTMABulkLoadR2(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 2);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR4
void CUDAIntrinsicLibrary::genTMABulkLoadR4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_LOADR8
void CUDAIntrinsicLibrary::genTMABulkLoadR8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkLoad(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                 fir::getBase(args[2]), fir::getBase(args[3]), eleSize);
}

// TMA_BULK_S2G
void CUDAIntrinsicLibrary::genTMABulkS2G(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value src = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[0]),
                                          aiir::NVVM::NVVMMemorySpace::Shared);
  aiir::Value dst = convertPtrToNVVMSpace(builder, loc, fir::getBase(args[1]),
                                          aiir::NVVM::NVVMMemorySpace::Global);
  aiir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(
      builder, loc, dst, src, fir::getBase(args[2]), {}, {});

  aiir::NVVM::InlinePtxOp::create(builder, loc, aiir::TypeRange{}, {}, {},
                                  "cp.async.bulk.commit_group;", {});
  aiir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc,
                                             builder.getI32IntegerAttr(0), {});
}

static void genTMABulkStore(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value src, aiir::Value dst, aiir::Value count,
                            aiir::Value eleSize) {
  aiir::Value size = aiir::arith::MulIOp::create(builder, loc, eleSize, count);
  setAlignment(src, kTMAAlignment);
  src = convertPtrToNVVMSpace(builder, loc, src,
                              aiir::NVVM::NVVMMemorySpace::Shared);
  dst = convertPtrToNVVMSpace(builder, loc, dst,
                              aiir::NVVM::NVVMMemorySpace::Global);
  aiir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(builder, loc, dst, src,
                                                     size, {}, {});
  aiir::NVVM::InlinePtxOp::create(builder, loc, aiir::TypeRange{}, {}, {},
                                  "cp.async.bulk.commit_group;", {});
  aiir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc,
                                             builder.getI32IntegerAttr(0), {});
}

// TMA_BULK_STORE_C4
void CUDAIntrinsicLibrary::genTMABulkStoreC4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_C8
void CUDAIntrinsicLibrary::genTMABulkStoreC8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 16);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_I4
void CUDAIntrinsicLibrary::genTMABulkStoreI4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_I8
void CUDAIntrinsicLibrary::genTMABulkStoreI8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R2
void CUDAIntrinsicLibrary::genTMABulkStoreR2(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 2);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R4
void CUDAIntrinsicLibrary::genTMABulkStoreR4(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 4);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_STORE_R8
void CUDAIntrinsicLibrary::genTMABulkStoreR8(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value eleSize =
      builder.createIntegerConstant(loc, builder.getI32Type(), 8);
  genTMABulkStore(builder, loc, fir::getBase(args[0]), fir::getBase(args[1]),
                  fir::getBase(args[2]), eleSize);
}

// TMA_BULK_WAIT_GROUP
void CUDAIntrinsicLibrary::genTMABulkWaitGroup(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  auto group = builder.getIntegerAttr(builder.getI32Type(), 0);
  aiir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, loc, group, {});
}

// ALL_SYNC, ANY_SYNC, BALLOT_SYNC
template <aiir::NVVM::VoteSyncKind kind>
aiir::Value
CUDAIntrinsicLibrary::genVoteSync(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::Value arg1 =
      fir::ConvertOp::create(builder, loc, builder.getI1Type(), args[1]);
  aiir::Type resTy = kind == aiir::NVVM::VoteSyncKind::ballot
                         ? builder.getI32Type()
                         : builder.getI1Type();
  auto voteRes =
      aiir::NVVM::VoteSyncOp::create(builder, loc, resTy, args[0], arg1, kind)
          .getResult();
  return fir::ConvertOp::create(builder, loc, resultType, voteRes);
}

} // namespace fir
