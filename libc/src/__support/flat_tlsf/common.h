//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide common definitions and constants for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_COMMON_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_COMMON_H

#include "src/__support/macros/config.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

using Byte = unsigned char;

constexpr size_t CHUNK_UNIT = 4 * sizeof(size_t);

constexpr size_t GAP_NODE_OFFSET = 0;
constexpr size_t GAP_BIN_OFFSET = sizeof(size_t) * 2;
constexpr size_t GAP_LOW_SIZE_OFFSET = sizeof(size_t) * 3;
constexpr size_t GAP_HIGH_SIZE_OFFSET = sizeof(size_t);

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_COMMON_H
