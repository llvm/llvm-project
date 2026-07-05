//===-- libc gwp asan crash test ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

int main() {
  char retval = 0;
  for (unsigned i = 0; i < 0x10000; ++i) {
    char *Ptr = reinterpret_cast<char *>(malloc(10));

    for (unsigned i = 0; i < 10; ++i) {
      *(Ptr + i) = 0x0;
    }

    free(Ptr);
    volatile char x = *Ptr;
    retval = retval + x;
  }
  return retval;
}
