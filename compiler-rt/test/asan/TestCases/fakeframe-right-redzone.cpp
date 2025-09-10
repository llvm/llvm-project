// RUN: %clangxx_asan -O0 %s -o %t && %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O0 %s -o %t -fsanitize-address-use-after-return=always && not %run %t 2>&1 | FileCheck %s

#include "defines.h"
#include <stdint.h>
#include <string.h>

#define kFrameSize (2048)
#define KFrameSizeMask (0x07ff)

ATTRIBUTE_NOINLINE
char *pretend_to_do_something(char *x) {
  __asm__ __volatile__("" : : "r"(x) : "memory");
  return x;
}

ATTRIBUTE_NOINLINE
char *OverwriteFakeFrameLastWord() {
  char x[1024];
  memset(x, 0, sizeof(x));
  uint64_t ptr_int = (reinterpret_cast<uint64_t>(x) & ~KFrameSizeMask) +
                     kFrameSize - sizeof(char **);
  char **ptr = reinterpret_cast<char **>(ptr_int);
  *ptr = nullptr;
  return pretend_to_do_something(x);
}

int main(int argc, char **argv) {
  char *x = OverwriteFakeFrameLastWord();
  // CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on address
  // CHECK: is located in stack of thread T0 at offset {{2040|2044}} in frame
  // CHECK:  in OverwriteFakeFrameLastWord{{.*}}fakeframe-right-redzone.cpp:
  // CHECK: [{{16|32}}, {{1040|1056}}) 'x'
  pretend_to_do_something(x);
  return 0;
}
