// RUN: %clangxx_asan %s -o %t && %run %t
// RUN: %clangxx_asan %s -mllvm -asan-use-after-return=runtime -o %t && env ASAN_OPTIONS=detect_stack_use_after_return=1 %run %t
// RUN: %clangxx_asan %s -mllvm -asan-use-after-return=always -o %t && %run %t

#include "defines.h"

#include <cassert>
#include <sanitizer/asan_interface.h>

volatile uintptr_t saved;

ATTRIBUTE_NOINLINE bool IsOnRealStack(uintptr_t caller_frame) {
  uintptr_t this_frame =
      reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
  return this_frame <= saved && saved <= caller_frame;
}

ATTRIBUTE_NOINLINE bool IsOnStack() {
  volatile char temp = ' ';
  saved = reinterpret_cast<uintptr_t>(&temp);
  return IsOnRealStack(reinterpret_cast<uintptr_t>(__builtin_frame_address(0)));
}

int main(int argc, char *argv[]) {
  assert(!IsOnStack());

  __asan_suppress_fake_stack();
  assert(IsOnStack());

  __asan_unsuppress_fake_stack();
  assert(!IsOnStack());

  return 0;
}
