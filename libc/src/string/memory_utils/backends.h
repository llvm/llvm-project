//===-- Elementary operations to compose memory primitives ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the concept of a Backend.
// It constitutes the lowest level of the framework and is akin to instruction
// selection. It defines how to implement aligned/unaligned,
// temporal/non-temporal native loads and stores for a particular architecture
// as well as efficient ways to fill and compare types.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKENDS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKENDS_H

#include "src/string/memory_utils/address.h" // Temporality, Aligned
#include "src/string/memory_utils/sized_op.h" // SizedOp
#include <stddef.h>                           // size_t
#include <stdint.h>                           // uint##_t

namespace __llvm_libc {

// Backends must implement the following interface.
struct NoBackend {
  static constexpr bool IS_BACKEND_TYPE = true;

  // Loads a T from `src` honoring Temporality and Alignment.
  template <typename T, Temporality, Aligned> static T load(const T *src);

  // Stores a T to `dst` honoring Temporality and Alignment.
  template <typename T, Temporality, Aligned>
  static void store(T *dst, T value);

  // Returns a T filled with `value` bytes.
  template <typename T> static T splat(ubyte value);

  // Returns zero iff v1 == v2.
  template <typename T> static uint64_t notEquals(T v1, T v2);

  // Returns zero iff v1 == v2, a negative number if v1 < v2 and a positive
  // number otherwise.
  template <typename T> static int32_t threeWayCmp(T v1, T v2);

  // Returns the type to use to consume Size bytes.
  // If no type handles Size bytes at once
  template <size_t Size> using getNextType = void;
};

} // namespace __llvm_libc

// We inline all backend implementations here to simplify the build system.
// Each file need to be guarded with the appropriate LLVM_LIBC_ARCH_XXX ifdef.
#include "src/string/memory_utils/backend_aarch64.h"
#include "src/string/memory_utils/backend_scalar.h"
#include "src/string/memory_utils/backend_x86.h"

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKENDS_H
