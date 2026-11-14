//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Parallel.h"
#include "llvm/Support/Parallel.h"
#include <algorithm>
#include <atomic>

using namespace llvm;

void lld::parallelForSharded(size_t numShards, size_t end,
                             function_ref<void(size_t, size_t)> fn) {
  std::atomic<size_t> next = 0;
  auto work = [&](size_t shard) {
    for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < end;)
      fn(shard, i);
  };
  // Run shard 0 on the calling thread.
  parallel::TaskGroup tg;
  for (size_t shard = 1, e = std::min(numShards, end); shard < e; ++shard)
    tg.spawn([&, shard] { work(shard); });
  work(0);
}

void lld::parallelForFineGrained(size_t end, function_ref<void(size_t)> fn) {
  parallelForSharded(parallel::getThreadCount(), end,
                     [&](size_t, size_t i) { fn(i); });
}
