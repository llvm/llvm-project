//===-- scudo_hooks_test.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "allocator_config.h"
#include "combined.h"

namespace {
void *LastAllocatedPtr = nullptr;
size_t LastRequestSize = 0;
void *LastDeallocatedPtr = nullptr;
} // namespace

// Scudo defines weak symbols that can be defined by a client binary
// to register callbacks at key points in the allocation timeline.  In
// order to enforce those invariants, we provide definitions that
// update some global state every time they are called, so that tests
// can inspect their effects.  An unfortunate side effect of this
// setup is that because those symbols are part of the binary, they
// can't be selectively enabled; that means that they will get called
// on unrelated tests in the same compilation unit. To mitigate this
// issue, we insulate those tests in a separate compilation unit.
extern "C" {
__attribute__((visibility("default"))) void __scudo_allocate_hook(void *Ptr,
                                                                  size_t Size) {
  LastAllocatedPtr = Ptr;
  LastRequestSize = Size;
}
__attribute__((visibility("default"))) void __scudo_deallocate_hook(void *Ptr) {
  LastDeallocatedPtr = Ptr;
}
}

// Simple check that allocation callbacks, when registered, are called:
//   1) __scudo_allocate_hook is called when allocating.
//   2) __scudo_deallocate_hook is called when deallocating.
//   3) Both hooks are called when reallocating.
//   4) Neither are called for a no-op reallocation.
TEST(ScudoHooksTest, AllocateHooks) {
  scudo::Allocator<scudo::DefaultConfig> Allocator;
  constexpr scudo::uptr DefaultSize = 16U;
  constexpr scudo::Chunk::Origin Origin = scudo::Chunk::Origin::Malloc;

  // Simple allocation and deallocation.
  {
    LastAllocatedPtr = nullptr;
    LastRequestSize = 0;

    void *Ptr = Allocator.allocate(DefaultSize, Origin);

    EXPECT_EQ(Ptr, LastAllocatedPtr);
    EXPECT_EQ(DefaultSize, LastRequestSize);

    LastDeallocatedPtr = nullptr;

    Allocator.deallocate(Ptr, Origin);

    EXPECT_EQ(Ptr, LastDeallocatedPtr);
  }

  // Simple no-op, same size reallocation.
  {
    void *Ptr = Allocator.allocate(DefaultSize, Origin);

    LastAllocatedPtr = nullptr;
    LastRequestSize = 0;
    LastDeallocatedPtr = nullptr;

    void *NewPtr = Allocator.reallocate(Ptr, DefaultSize);

    EXPECT_EQ(Ptr, NewPtr);
    EXPECT_EQ(nullptr, LastAllocatedPtr);
    EXPECT_EQ(0, LastRequestSize);
    EXPECT_EQ(nullptr, LastDeallocatedPtr);
  }

  // Reallocation in increasing size classes. This ensures that at
  // least one of the reallocations will be meaningful.
  {
    void *Ptr = Allocator.allocate(0, Origin);

    for (scudo::uptr ClassId = 1U;
         ClassId <= scudo::DefaultConfig::Primary::SizeClassMap::LargestClassId;
         ++ClassId) {
      const scudo::uptr Size =
          scudo::DefaultConfig::Primary::SizeClassMap::getSizeByClassId(
              ClassId);

      LastAllocatedPtr = nullptr;
      LastRequestSize = 0;
      LastDeallocatedPtr = nullptr;

      void *NewPtr = Allocator.reallocate(Ptr, Size);

      if (NewPtr != Ptr) {
        EXPECT_EQ(NewPtr, LastAllocatedPtr);
        EXPECT_EQ(Size, LastRequestSize);
        EXPECT_EQ(Ptr, LastDeallocatedPtr);
      } else {
        EXPECT_EQ(nullptr, LastAllocatedPtr);
        EXPECT_EQ(0, LastRequestSize);
        EXPECT_EQ(nullptr, LastDeallocatedPtr);
      }

      Ptr = NewPtr;
    }
  }
}
