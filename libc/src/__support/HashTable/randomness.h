//===-- HashTable Randomness ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_HASHTABLE_RANDOMNESS_H
#define LLVM_LIBC_SRC___SUPPORT_HASHTABLE_RANDOMNESS_H

#include "src/__support/common.h"
#include "src/__support/hash.h"
#include "src/__support/macros/attributes.h"
#if defined(LIBC_HASHTABLE_USE_GETRANDOM)
#include "src/errno/libc_errno.h"
#include "src/sys/random/getrandom.h"
#endif

namespace LIBC_NAMESPACE {
namespace internal {
namespace randomness {
// We need an initial state for the hash function. More entropy are to be added
// at the first use and each round of reseeding. The following random numbers
// are generated from https://www.random.org/cgi-bin/randbyte?nbytes=64&format=h
LIBC_INLINE_VAR thread_local HashState state = {
    0x38049a7ea6f5a79b, 0x45cb02147c3f718a, 0x53eb431c12770718,
    0x5b55742bd20a2fcb};
LIBC_INLINE_VAR thread_local uint64_t counter = 0;
LIBC_INLINE_VAR constexpr uint64_t RESEED_PERIOD = 1024;
LIBC_INLINE uint64_t next_random_seed() {
  if (counter % RESEED_PERIOD == 0) {
    uint64_t entropy[2];
    entropy[0] = reinterpret_cast<uint64_t>(&entropy);
    entropy[1] = reinterpret_cast<uint64_t>(&state);
#if defined(LIBC_HASHTABLE_USE_GETRANDOM)
    int errno_backup = libc_errno;
    ssize_t count = sizeof(entropy);
    uint8_t *buffer = reinterpret_cast<uint8_t *>(entropy);
    while (count > 0) {
      ssize_t len = getrandom(buffer, count, 0);
      if (len == -1) {
        if (libc_errno == ENOSYS)
          break;
        continue;
      }
      count -= len;
      buffer += len;
    }
    libc_errno = errno_backup;
#endif
    state.update(&entropy, sizeof(entropy));
  }
  state.update(&counter, sizeof(counter));
  counter++;
  return state.finish();
}

} // namespace randomness
} // namespace internal
} // namespace LIBC_NAMESPACE
#endif // LLVM_LIBC_SRC___SUPPORT_HASHTABLE_RANDOMNESS_H
