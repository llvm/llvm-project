// Check unsuppressing fake stack does not reenable it if disabled via compile or runtime options.
//
// RUN: %clangxx_asan %s -mllvm -asan-use-after-return=never -o %t && %run %t
// RUN: %clangxx_asan %s -mllvm -asan-use-after-return=runtime -o %t && env ASAN_OPTIONS=detect_stack_use_after_return=0 %run %t

#include "defines.h"

#include <cassert>
#include <sanitizer/asan_interface.h>

volatile uintptr_t saved;

ATTRIBUTE_NOINLINE bool IsOnRealStack(uintptr_t parent_frame,
                                      uintptr_t var_addr) {
  uintptr_t this_frame =
      reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
  return this_frame <= var_addr && var_addr <= parent_frame;
}

ATTRIBUTE_NOINLINE bool IsOnRealStack(uintptr_t parent_frame) {
  volatile char temp = ' ';
  saved = reinterpret_cast<uintptr_t>(&temp);
  return IsOnRealStack(parent_frame, saved);
}

ATTRIBUTE_NOINLINE bool IsOnRealStack() {
  return IsOnRealStack(reinterpret_cast<uintptr_t>(__builtin_frame_address(0)));
}

int main(int argc, char *argv[]) {
  assert(IsOnRealStack());

  __asan_suppress_fake_stack();
  assert(IsOnRealStack());

  __asan_unsuppress_fake_stack();
  assert(IsOnRealStack());

  return 0;
}
