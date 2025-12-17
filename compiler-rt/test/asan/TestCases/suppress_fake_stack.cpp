// This file tests the suppression of fake stack frames in ASan.

// We need to separate the MSVC and non-MSVC tests because MSVC because the MSVC enablement of 'stack-use-after-return' detection is different.

// MSVC tests
// RUN: %if target={{.*-windows-msvc.*}} %{ %clangxx_asan -fsanitize-address-use-after-return %s -o %t && env ASAN_OPTIONS=detect_stack_use_after_return=1 %run %t %}

// Non-MSVC tests
// RUN: %if !target={{.*-windows-msvc.*}} %{ %clangxx_asan %s -o %t && env ASAN_OPTIONS=detect_stack_use_after_return=1 %run %t %}
// RUN: %if !target={{.*-windows-msvc.*}} %{ %clangxx_asan %s -mllvm -asan-use-after-return=runtime -o %t && env ASAN_OPTIONS=detect_stack_use_after_return=1 %run %t %}
// RUN: %if !target={{.*-windows-msvc.*}} %{ %clangxx_asan %s -mllvm -asan-use-after-return=always -o %t && %run %t %}

#include "defines.h"

#include <cassert>
#include <sanitizer/asan_interface.h>

volatile uintptr_t saved;

// Use FRAME_ADDRESS macro to get the current frame address
#ifdef _MSC_VER
// MSVC does not have a 'builtin_frame_address' equivalent.
// However, for the purposes of this test, its `_AddressOfReturnAddress` built-in suffices
#  define FRAME_ADDRESS reinterpret_cast<uintptr_t>(_AddressOfReturnAddress())
#else
#  define FRAME_ADDRESS reinterpret_cast<uintptr_t>(__builtin_frame_address(0))
#endif

ATTRIBUTE_NOINLINE bool IsOnRealStack(uintptr_t parent_frame,
                                      uintptr_t var_addr) {
  uintptr_t this_frame = FRAME_ADDRESS;
  return this_frame <= var_addr && var_addr <= parent_frame;
}

ATTRIBUTE_NOINLINE bool IsOnRealStack(uintptr_t parent_frame) {
  volatile char temp = ' ';
  saved = reinterpret_cast<uintptr_t>(&temp);
  return IsOnRealStack(parent_frame, saved);
}

ATTRIBUTE_NOINLINE bool IsOnRealStack() { return IsOnRealStack(FRAME_ADDRESS); }

int main(int argc, char *argv[]) {
  assert(!IsOnRealStack());

  __asan_suppress_fake_stack();
  assert(IsOnRealStack());

  __asan_unsuppress_fake_stack();
  assert(!IsOnRealStack());

  return 0;
}
