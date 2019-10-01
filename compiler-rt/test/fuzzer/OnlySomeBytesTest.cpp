// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Find ABCxxFxUxZxxx... (2048+ bytes, 'x' is any byte)
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

const size_t N = 2048;
typedef const uint8_t *IN;

static volatile int one = 1;

extern "C" {
__attribute__((noinline)) void bad() {
  fprintf(stderr, "BINGO\n");
  if (one)
    abort();
}

__attribute__((noinline)) void f0(IN in) {
  uint32_t x = in[5] + 251 * in[7] + 251 * 251 * in[9];
  if (x == 'F' + 251 * 'U' + 251 * 251 * 'Z') {
    // artificially inflate uncovered control in f0
    // so that auto-focus is more likely to chose this function.
    if (one == -1) {
      if (one == 2) one = 1;    if (one == 3) one = 1;    if (one == 4) one = 1;
      if (one == 5) one = 1;    if (one == 6) one = 1;    if (one == 7) one = 1;
      if (one == 8) one = 1;    if (one == 9) one = 1;    if (one == 0) one = 1;
    }
    bad();
  }
}

__attribute__((noinline)) void fD(IN in) { f0(in); }
__attribute__((noinline)) void fC(IN in) { if (in[2] == 'C') fD(in); }
__attribute__((noinline)) void fB(IN in) { if (in[1] == 'B') fC(in); }
__attribute__((noinline)) void fA(IN in) { if (in[0] == 'A') fB(in); }

} // extern "C"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < N) return 0;
  fA((IN)Data);
  return 0;
}
