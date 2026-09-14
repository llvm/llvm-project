//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// InlineAny utility template type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_INLINE_ANY_H
#define LLVM_LIBC_SRC___SUPPORT_INLINE_ANY_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"

namespace LIBC_NAMESPACE_DECL {

// A utility providing inline storage for a type-erased value with trivial type.
//
// The maximum size and alignment of stored types is controlled by `Size` and
// `Alignment`. Does not store or check any type information for the saved value.
// Example:
//
//     InlineAny<20> any;
//     any.store(5.0);
//     double value = any.load<double>();
template <size_t Size, size_t Alignment = alignof(max_align_t)>
class InlineAny {
  alignas(Alignment) char buffer[Size];

public:
  template <typename T> LIBC_INLINE void store(const T &value) {
    static_assert(cpp::is_trivially_copyable_v<T> &&
                  cpp::is_trivially_constructible_v<T> &&
                  cpp::is_trivially_destructible_v<T> && sizeof(T) <= Size &&
                  alignof(T) <= Alignment);
    cpp::inline_copy<sizeof(T)>(reinterpret_cast<const char *>(&value), buffer);
  }

  template <typename T> LIBC_INLINE T load() const {
    static_assert(cpp::is_trivially_copyable_v<T> &&
                  cpp::is_trivially_constructible_v<T> &&
                  cpp::is_trivially_destructible_v<T> && sizeof(T) <= Size &&
                  alignof(T) <= Alignment);
    T result;
    cpp::inline_copy<sizeof(T)>(buffer, reinterpret_cast<char *>(&result));
    return result;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_INLINE_ANY_H
