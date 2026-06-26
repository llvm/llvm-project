//===-- flop_counter_runtime.cpp - FLOP Counter Runtime ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime for counting floating-point operations.
// It hooks into instrumentation points inserted by the LLVM Instrumentor pass.
//
//===----------------------------------------------------------------------===//

#include "../instrumentor_runtime.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

/// FLOP counter statistics (thread-safe using atomics)
struct FlopCounterStats {
  std::atomic<uint64_t> TotalFlops{0};
  std::atomic<uint64_t> FloatOps{0};  // 32-bit float operations
  std::atomic<uint64_t> DoubleOps{0}; // 64-bit double operations
  std::atomic<uint64_t> ExtendedOps{
      0}; // 80/128-bit extended precision operations
  std::atomic<uint64_t> VectorFlops{0}; // Total FLOPs from vector operations
  std::atomic<uint64_t> AddOps{0};
  std::atomic<uint64_t> MulOps{0};
  std::atomic<uint64_t> DivOps{0};
  std::atomic<uint64_t> FmaOps{0};   // Fused multiply-add operations
  std::atomic<uint64_t> OtherOps{0}; // sqrt, sin, cos, etc.
};

// Global statistics counters
static FlopCounterStats *Stats = nullptr;

enum {
  LLVMOpcodeFAdd = 15,
  LLVMOpcodeFSub = 17,
  LLVMOpcodeFMul = 19,
  LLVMOpcodeFDiv = 22,
  LLVMOpcodeFRem = 25,
  LLVMOpcodeFNeg = 13,
};

} // namespace

extern "C" {

__attribute__((constructor(1000))) void __flop_counter_initialize() {
  Stats = new FlopCounterStats();
}

__attribute__((destructor(1000))) void __flop_counter_finalize() {
  std::printf("\n");
  std::printf("=================================================\n");
  std::printf("           FLOP Counter Statistics\n");
  std::printf("=================================================\n");
  std::printf("Total FLOPs:              %20llu\n",
              Stats->TotalFlops.load(std::memory_order_relaxed));
  std::printf("\n");
  std::printf("By Precision:\n");
  std::printf("  Single (float):         %20llu\n",
              Stats->FloatOps.load(std::memory_order_relaxed));
  std::printf("  Double (double):        %20llu\n",
              Stats->DoubleOps.load(std::memory_order_relaxed));
  std::printf("  Extended (fp80/fp128):  %20llu\n",
              Stats->ExtendedOps.load(std::memory_order_relaxed));
  std::printf("  Vector FLOPs:           %20llu\n",
              Stats->VectorFlops.load(std::memory_order_relaxed));
  std::printf("\n");
  std::printf("By Operation:\n");
  std::printf("  Addition/Subtraction:   %20llu\n",
              Stats->AddOps.load(std::memory_order_relaxed));
  std::printf("  Multiplication:         %20llu\n",
              Stats->MulOps.load(std::memory_order_relaxed));
  std::printf("  Division:               %20llu\n",
              Stats->DivOps.load(std::memory_order_relaxed));
  std::printf("  Fused Multiply-Add:     %20llu\n",
              Stats->FmaOps.load(std::memory_order_relaxed));
  std::printf("  Other (sqrt, sin, ...): %20llu\n",
              Stats->OtherOps.load(std::memory_order_relaxed));
  std::printf("=================================================\n");

  delete Stats;
}

void __flop_counter_post_numeric(int32_t TypeId, int32_t SubTypeId,
                                 int32_t Size, int32_t Opcode) {
  bool IsVector = false;
  switch (TypeId) {
  case FixedVectorTyID:
  case ScalableVectorTyID:
    IsVector = true;
    TypeId = SubTypeId;
    break;
  default:
    break;
  };

  int32_t TypeSize = Size;
  switch (TypeId) {
  case HalfTyID:
  case BFloatTyID:
    TypeSize = 2;
    break;
  case FloatTyID:
    TypeSize = 4;
    break;
  case DoubleTyID:
    TypeSize = 8;
    break;
  case X86_FP80TyID:
  case FP128TyID:
  case PPC_FP128TyID:
    TypeSize = 16;
    break;
  default:
    break;
  };

  // Determine FLOP count based on whether it's a vector operation
  uint64_t FlopCount = Size / TypeSize;
  if (IsVector) {
    Stats->VectorFlops.fetch_add(FlopCount, std::memory_order_relaxed);
  } else {
    // Categorize by precision
    if (TypeId == 2) {
      Stats->FloatOps.fetch_add(1, std::memory_order_relaxed);
    } else if (TypeId == 3) {
      Stats->DoubleOps.fetch_add(1, std::memory_order_relaxed);
    } else {
      Stats->ExtendedOps.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Categorize by operation type
  switch (Opcode) {
  case LLVMOpcodeFAdd:
  case LLVMOpcodeFSub:
    Stats->AddOps.fetch_add(FlopCount, std::memory_order_relaxed);
    break;
  case LLVMOpcodeFMul:
    Stats->MulOps.fetch_add(FlopCount, std::memory_order_relaxed);
    break;
  case LLVMOpcodeFDiv:
  case LLVMOpcodeFRem:
    Stats->DivOps.fetch_add(FlopCount, std::memory_order_relaxed);
    break;
  default:
    Stats->OtherOps.fetch_add(FlopCount, std::memory_order_relaxed);
    break;
  }

  Stats->TotalFlops.fetch_add(FlopCount, std::memory_order_relaxed);
}

} // extern "C"
