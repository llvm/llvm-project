//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H

#include <__config>

// Combined OpenMP CPU and GPU Backend
// ===================================
// Contrary to the CPU backends found in ./cpu_backends/, the OpenMP backend can
// target both CPUs and GPUs. The OpenMP standard defines that when offloading
// code to an accelerator, the compiler must generate a fallback code for
// execution on the host. Thereby, the backend works as a CPU backend if no
// targeted accelerator is available at execution time. The target regions can
// also be compiled directly for a CPU architecture, for instance by adding the
// command-line option `-fopenmp-targets=x86_64-pc-linux-gnu` in Clang.
//
// When is an Algorithm Offloaded?
// -------------------------------
// Only parallel algorithms with the parallel unsequenced execution policy are
// offloaded to the device. We cannot offload parallel algorithms with a
// parallel execution policy to GPUs because invocations executing in the same
// thread "are indeterminately sequenced with respect to each other" which we
// cannot guarantee on a GPU.
//
// The standard draft states that "the semantics [...] allow the implementation
// to fall back to sequential execution if the system cannot parallelize an
// algorithm invocation". If it is not deemed safe to offload the parallel
// algorithm to the device, we first fall back to a parallel unsequenced
// implementation from ./cpu_backends. The CPU implementation may then fall back
// to sequential execution. In that way we strive to achieve the best possible
// performance.
//
// Further, "it is the caller's responsibility to ensure that the invocation
// does not introduce data races or deadlocks."
//
// Implicit Assumptions
// --------------------
// If the user provides a function pointer as an argument to a parallel
// algorithm, it is assumed that it is the device pointer as there is currently
// no way to check whether a host or device pointer was passed.
//
// Mapping Clauses
// ---------------
// In some of the parallel algorithms, the user is allowed to provide the same
// iterator as input and output. The order of the maps matters because OpenMP
// keeps a reference counter of which variables have been mapped to the device.
// Thereby, a varible is only copied to the device if its reference counter is
// incremented from zero, and it is only copied back to the host when the
// reference counter is decremented to zero again.
// This allows nesting mapped regions, for instance in recursive functions,
// without enforcing a lot of unnecessary data movement.
// Therefore, `pragma omp target data map(to:...)` must be used before
// `pragma omp target data map(alloc:...)`. Conversely, the maps with map
// modifier `release` must be placed before the maps with map modifier `from`
// when transferring the result from the device to the host.
//
// Example: Assume `a` and `b` are pointers to the same array.
// ``` C++
// #pragma omp target enter data map(alloc:a[0:n])
// // The reference counter is incremented from 0 to 1. a is not copied to the
// // device because of the `alloc` map modifier.
// #pragma omp target enter data map(to:b[0:n])
// // The reference counter is incremented from 1 to 2. b is not copied because
// // the reference counter is positive. Therefore b, and a, are uninitialized
// // on the device.
// ```
//
// Exceptions
// ----------
// Currently, GPU architectures do not handle exceptions. OpenMP target regions
// are allowed to contain try/catch statements and throw expressions in Clang,
// but if a throw expression is reached, it will terminate the program. That
// does not conform to the C++ standard.
//
// [This document](https://eel.is/c++draft/algorithms.parallel) has been used as
// reference for these considerations.

#include <__algorithm/pstl_backends/openmp/any_of.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/pstl_backends/openmp/fill.h>
#include <__algorithm/pstl_backends/openmp/find_if.h>
#include <__algorithm/pstl_backends/openmp/for_each.h>
#include <__algorithm/pstl_backends/openmp/merge.h>
#include <__algorithm/pstl_backends/openmp/stable_sort.h>
#include <__algorithm/pstl_backends/openmp/transform.h>
#include <__algorithm/pstl_backends/openmp/transform_reduce.h>

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H
