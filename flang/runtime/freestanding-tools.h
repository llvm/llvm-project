//===-- runtime/freestanding-tools.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
#define FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_

#include "flang/Runtime/api-attrs.h"
#include <algorithm>

// The file defines a set of utilities/classes that might be
// used to get reduce the dependency on external libraries (e.g. libstdc++).

#if !defined(STD_FILL_N_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_FILL_N_UNSUPPORTED 1
#endif

namespace Fortran::runtime {

#if STD_FILL_N_UNSUPPORTED
// Provides alternative implementation for std::fill_n(), if
// it is not supported.
template <typename A>
static inline RT_API_ATTRS void fill_n(
    A *start, std::size_t count, const A &value) {
#if STD_FILL_N_UNSUPPORTED
  for (std::size_t j{0}; j < count; ++j)
    start[j] = value;
#else
  std::fill_n(start, count, value);
#endif
}
#else // !STD_FILL_N_UNSUPPORTED
using std::fill_n;
#endif // !STD_FILL_N_UNSUPPORTED

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
