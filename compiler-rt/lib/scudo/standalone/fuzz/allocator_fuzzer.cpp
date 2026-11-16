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
#include <format>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  FuzzedDataProvider FDP(Data, Size);
  if (FDP.remaining_bytes() == 0)
    return 0;

  // TODO: Make this configurable.
  static bool ZeroFill = true;

  // After a number of operation, reset the the allocator with different
  // parameters.
  using AllocatorT = scudo::Allocator<scudo::Config>;
  static std::unique_ptr<AllocatorT> Allocator;
  if (Allocator == nullptr) {
    Allocator.reset(new AllocatorT());
    Allocator->init();
    // The way we are using the allocator doesn't work properly with MTE
    // enabled.
    if (scudo::systemSupportsMemoryTagging())
      Allocator->disableMemoryTagging();
    Allocator->setOption(scudo::Option::ReleaseInterval, 1000);

    // Set the fill type once for the allocator, changing the fill type during
    // the run can lead to some data not being zeroed if zero on deallocate
    // is enabled.
    std::cout << std::format("Allocator Created with ZeroFill {}\n", ZeroFill);
    Allocator->setFillContents(ZeroFill ? scudo::ZeroFill : scudo::NoFill);
  }

  constexpr size_t kMaxAllocatedBytes = 50 * 1024 * 1024;
  size_t TotalAllocatedBytes = 0;
  std::vector<void *> Allocations;
  while (FDP.remaining_bytes() > 0) {
    uint8_t Op = FDP.ConsumeIntegralInRange<uint8_t>(0, 4);
    if (Op == 0 || Op == 1) {
      if (TotalAllocatedBytes >= kMaxAllocatedBytes)
        continue;

      bool ZeroContents = FDP.ConsumeBool();
      size_t ReqSize =
          FDP.ConsumeIntegralInRange<size_t>(1, 1 << 20); // Up to 1MB
      void *Ptr;
      if (Op == 0) {
        // Allocate default alignment
        Ptr = Allocator->allocate(ReqSize, scudo::Chunk::Origin::Malloc, 2,
                                  ZeroContents);
      } else {
        // Allocate with alignment
        size_t Alignment =
            1 << FDP.ConsumeIntegralInRange<size_t>(4, 12); // 16 to 4096
        Ptr = Allocator->allocate(ReqSize, scudo::Chunk::Origin::Memalign,
                                  Alignment, ZeroContents);
        CHECK_EQ(0, reinterpret_cast<uintptr_t>(Ptr) & (Alignment - 1));
      }
      CHECK(Ptr != nullptr);
      size_t Size = Allocator->getUsableSize(Ptr);
      TotalAllocatedBytes += Size;
      Allocations.push_back(Ptr);
      if (ZeroContents || ZeroFill) {
        bool Mismatch = false;
        uint8_t *Data = reinterpret_cast<uint8_t *>(Ptr);
        for (size_t i = 0; i < Size; i++) {
          if (Data[i] != 0) {
            if (!Mismatch) {
              std::cerr << std::format(
                  "Failed Ptr {} Size {} ZeroContents {} ZeroFill {}\n", Ptr,
                  Size, ZeroContents, ZeroFill);
              Mismatch = true;
            }
            std::cerr << std::format("  Index {} Ptr {} Value {}\n", i,
                                     (void *)&Data[i], Data[i]);
          }
        }
        if (Mismatch)
          abort();
      }
      memset(Ptr, 0xff, Size);
    } else if (Op == 2) {
      if (Allocations.empty())
        continue;

      // Deallocate
      size_t Index =
          FDP.ConsumeIntegralInRange<size_t>(0, Allocations.size() - 1);
      TotalAllocatedBytes -= Allocator->getUsableSize(Allocations[Index]);
      Allocator->deallocate(Allocations[Index], scudo::Chunk::Origin::Malloc);
      Allocations.erase(Allocations.begin() + Index);
    } else if (Op == 3 && !Allocations.empty()) {
      // Reallocate (Assumes reallocate of a memalign does not crash).
      size_t Index =
          FDP.ConsumeIntegralInRange<size_t>(0, Allocations.size() - 1);
      size_t OldSize = Allocator->getUsableSize(Allocations[Index]);
      TotalAllocatedBytes -= OldSize;
      size_t NewSize = FDP.ConsumeIntegralInRange<size_t>(1, 1 << 20);
      void *NewPtr = Allocator->reallocate(Allocations[Index], NewSize);
      if (NewSize == 0) {
        CHECK(NewPtr == nullptr);
        Allocations.erase(Allocations.begin() + Index);
      } else {
        CHECK(NewPtr != nullptr);
        size_t Size = Allocator->getUsableSize(NewPtr);
        memset(NewPtr, 0xff, Size);
        Allocations[Index] = NewPtr;
        TotalAllocatedBytes -= Size;
      }
    } else if (Op == 4) {
      // ReleaseToOS
      scudo::ReleaseToOS ReleaseType =
          static_cast<scudo::ReleaseToOS>(FDP.ConsumeIntegralInRange<size_t>(
              0, static_cast<size_t>(scudo::ReleaseToOS::Last)));
      Allocator->releaseToOS(ReleaseType);
    }
  }

  for (void *Ptr : Allocations) {
    Allocator->deallocate(Ptr, scudo::Chunk::Origin::Malloc);
  }

  return 0;
}
