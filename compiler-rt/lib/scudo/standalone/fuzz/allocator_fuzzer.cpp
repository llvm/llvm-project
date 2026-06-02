//===-- allocator_fuzzer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SCUDO_FUZZ
#include "allocator_config.h"
#include "combined.h"
#include <fuzzer/FuzzedDataProvider.h>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  using AllocatorT = scudo::Allocator<scudo::Config>;
  static AllocatorT *Instance = []() {
    auto *A = new AllocatorT();
    A->init();
    // The way we are using the allocator doesn't work properly with MTE
    // enabled.
    if (scudo::systemSupportsMemoryTagging())
      A->disableMemoryTagging();
    return A;
  }();

  FuzzedDataProvider FDP(Data, Size);
  std::vector<void *> Allocations;

  Instance->setOption(scudo::Option::ReleaseInterval, 1000);
  constexpr size_t kMaxAllocatedBytes = 50 * 1024 * 1024;
  size_t TotalAllocatedBytes = 0;
  while (FDP.remaining_bytes() > 0) {
    uint8_t Op = FDP.ConsumeIntegralInRange<uint8_t>(0, 4);
    if ((Op == 0 || Op == 1) && TotalAllocatedBytes < kMaxAllocatedBytes) {
      size_t ReqSize =
          FDP.ConsumeIntegralInRange<size_t>(1, 1 << 20); // Up to 1MB
      void *Ptr;
      if (Op == 0) {
        // Allocate no alignment
        Ptr = Instance->allocate(ReqSize, scudo::Chunk::Origin::Malloc);
      } else {
        // Allocate with alignment
        size_t Alignment =
            1 << FDP.ConsumeIntegralInRange<size_t>(4, 12); // 16 to 4096
        Ptr = Instance->allocate(ReqSize, scudo::Chunk::Origin::Memalign,
                                 Alignment);
        CHECK_EQ(0, reinterpret_cast<uintptr_t>(Ptr) & (Alignment - 1));
      }
      CHECK(Ptr != nullptr);
      size_t Size = Instance->getUsableSize(Ptr);
      TotalAllocatedBytes += Size;
      Allocations.push_back(Ptr);
      memset(Ptr, 0xff, Size);
    } else if (Op == 2 && !Allocations.empty()) {
      // Deallocate
      size_t Index =
          FDP.ConsumeIntegralInRange<size_t>(0, Allocations.size() - 1);
      TotalAllocatedBytes -= Instance->getUsableSize(Allocations[Index]);
      Instance->deallocate(Allocations[Index], scudo::Chunk::Origin::Malloc);
      Allocations.erase(Allocations.begin() + Index);
    } else if (Op == 3 && !Allocations.empty()) {
      // Reallocate (Assumes reallocate of a memalign does not crash).
      size_t Index =
          FDP.ConsumeIntegralInRange<size_t>(0, Allocations.size() - 1);
      size_t OldSize = Instance->getUsableSize(Allocations[Index]);
      TotalAllocatedBytes -= OldSize;
      size_t NewSize = FDP.ConsumeIntegralInRange<size_t>(1, 1 << 20);
      void *NewPtr = Instance->reallocate(Allocations[Index], NewSize);
      if (NewSize == 0) {
        CHECK(NewPtr == nullptr);
        Allocations.erase(Allocations.begin() + Index);
      } else {
        CHECK(NewPtr != nullptr);
        size_t Size = Instance->getUsableSize(NewPtr);
        memset(NewPtr, 0xff, Size);
        Allocations[Index] = NewPtr;
        TotalAllocatedBytes -= Size;
      }
    } else if (Op == 4) {
      // ReleaseToOS
      scudo::ReleaseToOS ReleaseType =
          static_cast<scudo::ReleaseToOS>(FDP.ConsumeIntegralInRange<size_t>(
              0, static_cast<size_t>(scudo::ReleaseToOS::Last)));
      Instance->releaseToOS(ReleaseType);
    }
  }

  // Cleanup remaining
  for (void *Ptr : Allocations) {
    Instance->deallocate(Ptr, scudo::Chunk::Origin::Malloc);
  }

  return 0;
}
