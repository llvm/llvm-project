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

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_FORCE_LOCK_BASED_ATOMIC_SHARED_PTR
// ADDITIONAL_COMPILE_FLAGS(target=x86_64-unknown-linux-gnu): -march=x86-64-v2

// Same benches as atomic_shared_ptr.bench.cpp, but forces libc++'s stolen-bit
// spinlock path even on DWCAS targets. Same-machine A/B only; names match so
// compare-benchmarks can pair the two files. Driver:
//   python3 libcxx/test/benchmarks/atomic_shared_ptr_bench.runner.py --build <build>

#include "atomic_shared_ptr_bench.h"

BENCHMARK_MAIN();
