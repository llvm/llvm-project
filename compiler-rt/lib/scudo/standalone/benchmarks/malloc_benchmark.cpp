//===-- malloc_benchmark.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocator_config.h"
#include "combined.h"
#include "common.h"

#include "benchmark/benchmark.h"

#include <memory>

template <typename Config> static void BM_malloc_free(benchmark::State &State) {
  using AllocatorT = scudo::Allocator<Config>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();

  const size_t NBytes = State.range(0);
  size_t PageSize = scudo::getPageSizeCached();

  for (auto _ : State) {
    void *Ptr = Allocator->allocate(NBytes, scudo::Chunk::Origin::Malloc);
    auto *Data = reinterpret_cast<uint8_t *>(Ptr);
    for (size_t I = 0; I < NBytes; I += PageSize)
      Data[I] = 1;
    benchmark::DoNotOptimize(Ptr);
    Allocator->deallocate(Ptr, scudo::Chunk::Origin::Malloc);
  }

  State.SetBytesProcessed(uint64_t(State.iterations()) * uint64_t(NBytes));
}

static const size_t MinSize = 8;
static const size_t MaxSize = 128 * 1024;

// FIXME: Add DefaultConfig here once we can tear down the exclusive TSD
// cleanly.
BENCHMARK_TEMPLATE(BM_malloc_free, scudo::AndroidConfig)
    ->Range(MinSize, MaxSize);
BENCHMARK_TEMPLATE(BM_malloc_free, scudo::AndroidSvelteConfig)
    ->Range(MinSize, MaxSize);
#if SCUDO_CAN_USE_PRIMARY64
BENCHMARK_TEMPLATE(BM_malloc_free, scudo::FuchsiaConfig)
    ->Range(MinSize, MaxSize);
#endif

template <typename Config>
static void BM_malloc_free_loop(benchmark::State &State) {
  using AllocatorT = scudo::Allocator<Config>;
  auto Deleter = [](AllocatorT *A) {
    A->unmapTestOnly();
    delete A;
  };
  std::unique_ptr<AllocatorT, decltype(Deleter)> Allocator(new AllocatorT,
                                                           Deleter);
  Allocator->reset();

  const size_t NumIters = State.range(0);
  size_t PageSize = scudo::getPageSizeCached();
  void *Ptrs[NumIters];

  for (auto _ : State) {
    for (void *&Ptr : Ptrs) {
      Ptr = Allocator->allocate(8192, scudo::Chunk::Origin::Malloc);
      auto *Data = reinterpret_cast<uint8_t *>(Ptr);
      for (size_t I = 0; I < 8192; I += PageSize)
        Data[I] = 1;
      benchmark::DoNotOptimize(Ptr);
    }
    for (void *&Ptr : Ptrs)
      Allocator->deallocate(Ptr, scudo::Chunk::Origin::Malloc);
  }

  State.SetBytesProcessed(uint64_t(State.iterations()) * uint64_t(NumIters) *
                          8192);
}

static const size_t MinIters = 8;
static const size_t MaxIters = 32 * 1024;

// FIXME: Add DefaultConfig here once we can tear down the exclusive TSD
// cleanly.
BENCHMARK_TEMPLATE(BM_malloc_free_loop, scudo::AndroidConfig)
    ->Range(MinIters, MaxIters);
BENCHMARK_TEMPLATE(BM_malloc_free_loop, scudo::AndroidSvelteConfig)
    ->Range(MinIters, MaxIters);
#if SCUDO_CAN_USE_PRIMARY64
BENCHMARK_TEMPLATE(BM_malloc_free_loop, scudo::FuchsiaConfig)
    ->Range(MinIters, MaxIters);
#endif

BENCHMARK_MAIN();
