//===-- Common header for multiply-add implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_MULTIPLY_ADD_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_MULTIPLY_ADD_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA

namespace LIBC_NAMESPACE {
namespace fputil {

// Implement a simple wrapper for multiply-add operation:
//   multiply_add(x, y, z) = x*y + z
// which uses FMA instructions to speed up if available.

template <typename T>
LIBC_INLINE cpp::enable_if_t<(sizeof(T) > sizeof(void *)), T>
multiply_add(const T &x, const T &y, const T &z) {
  return x * y + z;
}

template <typename T>
LIBC_INLINE cpp::enable_if_t<(sizeof(T) <= sizeof(void *)), T>
multiply_add(T x, T y, T z) {
  return x * y + z;
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#if defined(LIBC_TARGET_CPU_HAS_FMA)

// FMA instructions are available.
#include "FMA.h"

namespace LIBC_NAMESPACE {
namespace fputil {

LIBC_INLINE float multiply_add(float x, float y, float z) {
  return fma(x, y, z);
}

LIBC_INLINE double multiply_add(double x, double y, double z) {
  return fma(x, y, z);
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LIBC_TARGET_CPU_HAS_FMA

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_MULTIPLY_ADD_H
