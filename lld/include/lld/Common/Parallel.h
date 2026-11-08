//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PARALLEL_H
#define LLD_PARALLEL_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>

namespace lld {
// Call fn(i) for each i in [0, n) in parallel, claiming one index at a time.
// Unlike llvm::parallelFor, which claims fixed chunks, this balances load when
// per-item cost varies widely.
void parallelForFineGrained(size_t n, llvm::function_ref<void(size_t)> fn);

// Like parallelForFineGrained, but run up to numShards workers and pass the
// worker's shard index, which is less than numShards, as the first argument.
void parallelForSharded(size_t numShards, size_t n,
                        llvm::function_ref<void(size_t, size_t)> fn);
} // namespace lld

#endif
