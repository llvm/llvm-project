//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// REQUIRES: stdlib=libc++
// UNSUPPORTED: no-threads
// Clang's x86-64 baseline does not define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16;
// without this, the "lock-free" file compiles the stolen-bit path. Same flag
// as atomic_shared_ptr_dwcas.pass.cpp. Keep in sync with the lock-based file
// so the A/B differs only by _LIBCPP_FORCE_LOCK_BASED_ATOMIC_SHARED_PTR.
// ADDITIONAL_COMPILE_FLAGS(target=x86_64-unknown-linux-gnu): -march=x86-64-v2

// libc++ default path: lock-free on x86_64/cx16 and AArch64/LSE, otherwise
// lock-based. Pair with atomic_shared_ptr_lock_based.bench.cpp on the same
// machine. Contended thread counts default to 2,4,8; override at process
// start with ATOMIC_SP_BENCH_THREADS=2,4,6,8.
//
// Same-machine A/B (each file separately - names collide if consolidated
// together). Driver next to these benches:
//   python3 libcxx/test/benchmarks/atomic_shared_ptr_bench.runner.py --build <build>
// Add --threads 2,4,6,8,10,12,14,16,18,20 to override the default 2,4,8.
// Needs: pip install -r libcxx/utils/requirements.txt
//
// Report ratios to std::atomic<uint64_t>::* in the same JSON, not raw ns/op.

#include "atomic_shared_ptr_bench.h"

BENCHMARK_MAIN();
