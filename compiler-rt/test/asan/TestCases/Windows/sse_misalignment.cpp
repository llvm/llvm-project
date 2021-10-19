// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %env_asan_opts=handle_sigfpe=1 not %run %t 2>&1 | FileCheck %s

// Test the error output from misaligned SSE2 memory access. This is a READ
// memory access. Windows appears to always provide an address of -1 for these
// types of faults, and there doesn't seem to be a way to distinguish them from
// other types of access violations without disassembling.

#include <emmintrin.h>
#include <stdio.h>

__m128i test() {
  char buffer[17] = {};
  __m128i a = _mm_load_si128((__m128i *)buffer);
  __m128i b = _mm_load_si128((__m128i *)(&buffer[0] + 1));
  return _mm_or_si128(a, b);
}

int main() {
  puts("before alignment fault");
  fflush(stdout);
  volatile __m128i v = test();
  return 0;
}
// CHECK: before alignment fault
// CHECK: ERROR: AddressSanitizer: access-violation on unknown address {{0x[fF]*}}
// CHECK-NEXT: The signal is caused by a READ memory access.
// CHECK-NEXT: #0 {{.*}} in test(void) {{.*}}misalignment.cpp:{{.*}}
