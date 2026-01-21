//===-- lf_rtl.cpp - LowFat Sanitizer Runtime Library ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the main file of the LowFat Sanitizer runtime library.
//
// LowFat pointers encode allocation bounds information directly in the pointer
// value through careful memory layout. This allows O(1) bounds checking without
// maintaining separate metadata.
//
//===----------------------------------------------------------------------===//

#include "lf_interface.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __sanitizer;

namespace __lowfat {

// Flag to track initialization state
static bool lowfat_inited = false;

// TODO: Add LowFat-specific configuration tables
// These will define the memory regions and size classes for LowFat allocations

static void PrintErrorAndDie(uptr ptr, uptr base, uptr bound) {
  Printf("=================================================================\n");
  Printf("==ERROR: LowFat: out-of-bounds access detected\n");
  Printf("  pointer: 0x%zx\n", ptr);
  Printf("  base:    0x%zx\n", base);
  Printf("  bound:   %zu bytes\n", bound);
  Printf("=================================================================\n");

  // Print stack trace
  BufferedStackTrace stack;
  stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), nullptr,
               common_flags()->fast_unwind_on_fatal);
  stack.Print();

  Die();
}

}  // namespace __lowfat

// ---------------------- Interface Functions ----------------------

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_init() {
  if (__lowfat::lowfat_inited)
    return;

  // Initialize sanitizer common
  // InitializeCommonFlags() would go here if we had custom flags

  Printf("LowFat Sanitizer: runtime initialized\n");

  __lowfat::lowfat_inited = true;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_check_bounds(uptr ptr, uptr size) {
  // TODO: Implement actual bounds checking
  // For now, this is a stub that does nothing
  //
  // The full implementation should:
  // 1. Extract the region index from the pointer's high bits
  // 2. Look up the allocation size for that region
  // 3. Compute the base address using the region's alignment
  // 4. Check if ptr + size <= base + allocation_size
  (void)ptr;
  (void)size;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lf_report_oob(uptr ptr, uptr base, uptr bound) {
  __lowfat::PrintErrorAndDie(ptr, base, bound);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_base(uptr ptr) {
  // TODO: Implement base address extraction
  // This will use the LowFat memory layout to compute the base
  (void)ptr;
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __lf_get_size(uptr ptr) {
  // TODO: Implement size extraction
  // This will look up the size class from the pointer's region
  (void)ptr;
  return 0;
}

}  // extern "C"

// Ensure initialization runs early via .preinit_array on ELF platforms
#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static auto preinit =
    __lf_init;
#endif