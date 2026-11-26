// The compiler keeps a magic number in the first word of an instrumented
// frame, inside its left redzone. To describe an address on the real stack,
// the runtime looks for the nearest left redzone below the address and expects
// that number there. Check that the report is complete if it is not there.
// Here code that is not instrumented overwrites it with the value in argv[1].

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=0 not %run %t 0 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,REAL
// 0x45E0360E is the number that a frame holds while it is being torn down.
// RUN: %env_asan_opts=detect_stack_use_after_return=0 not %run %t 0x45E0360E \
// RUN:   2>&1 | FileCheck %s --check-prefixes=CHECK,REAL
// A frame on the fake stack is described without that number, so nothing
// changes for it.
// RUN: %env_asan_opts=detect_stack_use_after_return=1 not %run %t 0 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,FAKE

// The place of the frame's first word depends on the shadow granularity and
// on the compiler.
// REQUIRES: shadow-scale-3
// UNSUPPORTED: MSVC

#include <sanitizer/asan_interface.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((no_sanitize("address"), noinline)) static void
Overwrite(char *p, uintptr_t value) {
  *(volatile uintptr_t *)p = value;
}

__attribute__((noinline)) static void Locate(char *p) {
  char name[8];
  void *region_address;
  size_t region_size;
  fprintf(stderr, "kind: %s\n",
          __asan_locate_address(p, name, sizeof(name), &region_address,
                                &region_size));
  fflush(stderr);
}

char *volatile sink;

int main(int argc, char **argv) {
  char buf[16];
  sink = buf;
  // The frame starts four words below its first variable.
  Overwrite(sink - 4 * sizeof(uintptr_t), (uintptr_t)strtoull(argv[1], 0, 0));
  Locate(sink);
  // CHECK: kind: stack
  return sink[16];
  // CHECK: ERROR: AddressSanitizer: stack-buffer-overflow
  // REAL: is located in stack of thread T0{{$}}
  // FAKE: is located in stack of thread T0 at offset {{[0-9]+}} in frame
  // FAKE: 'buf'{{.*}} <== Memory access at offset {{[0-9]+}} overflows this variable
  // CHECK: SUMMARY: AddressSanitizer: stack-buffer-overflow
}
