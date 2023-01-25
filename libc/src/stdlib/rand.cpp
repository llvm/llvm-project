//===-- Implementation of rand --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "src/__support/common.h"
#include "src/stdlib/rand_util.h"

namespace __llvm_libc {

// This rand function is the example implementation from the C standard. It is
// not cryptographically secure.
LLVM_LIBC_FUNCTION(int, rand, (void)) { // RAND_MAX is assumed to be 32767
  rand_next = rand_next * 1103515245 + 12345;
  return static_cast<unsigned int>((rand_next / 65536) % 32768);
}

} // namespace __llvm_libc
