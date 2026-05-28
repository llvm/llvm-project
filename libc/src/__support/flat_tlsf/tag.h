//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide tag utilities for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_TAG_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_TAG_H

#include "src/__support/flat_tlsf/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {
namespace tag {

static constexpr Byte ALLOCATED_FLAG = 0b0001;
static constexpr Byte ABOVE_FREE_FLAG = 0b0010;
static constexpr Byte HEAP_BASE_FLAG = 0b0100;
static constexpr Byte HEAP_END_FLAG = 0b1000;

LIBC_INLINE bool is_above_free(Byte tag) { return tag & ABOVE_FREE_FLAG; }

LIBC_INLINE bool is_allocated(Byte tag) { return tag & ALLOCATED_FLAG; }

LIBC_INLINE bool is_heap_base(Byte tag) { return tag & HEAP_BASE_FLAG; }

LIBC_INLINE bool is_heap_end(Byte tag) { return tag & HEAP_END_FLAG; }

LIBC_INLINE void set_above_free(Byte *ptr) { *ptr |= ABOVE_FREE_FLAG; }

LIBC_INLINE void clear_above_free(Byte *ptr) { *ptr &= ~ABOVE_FREE_FLAG; }

LIBC_INLINE void set_end_flag(Byte *ptr) { *ptr |= HEAP_END_FLAG; }

LIBC_INLINE void clear_end_flag(Byte *ptr) { *ptr &= ~HEAP_END_FLAG; }

} // namespace tag
} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_TAG_H
