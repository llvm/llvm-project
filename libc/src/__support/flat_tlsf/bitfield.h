//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide BitField class for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BITFIELD_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BITFIELD_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/flat_tlsf/bit_utils.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

struct alignas(16) BitField {
  static constexpr size_t BITS_PER_ELEMENT =
      cpp::numeric_limits<size_t>::digits;
  static constexpr size_t NUMBER_OF_ELEMENTS = 3;
  static constexpr size_t BITS = BITS_PER_ELEMENT * NUMBER_OF_ELEMENTS;

  cpp::array<size_t, NUMBER_OF_ELEMENTS> storage;

  LIBC_INLINE static constexpr BitField zeros() { return {}; }

  LIBC_INLINE constexpr uint32_t bit_scan_after(uint32_t bit) const {
    uint32_t array_index = bit / BITS_PER_ELEMENT;
    uint32_t element_index = bit % BITS_PER_ELEMENT;
    uint32_t bit_index =
        bit_utils::bit_scan_after(storage[array_index], element_index);
    if (bit_index < BITS_PER_ELEMENT)
      return array_index * BITS_PER_ELEMENT + bit_index;
    for (array_index = array_index + 1; array_index < NUMBER_OF_ELEMENTS;
         ++array_index) {
      bit_index = bit_utils::bit_scan_after(storage[array_index], 0);
      if (bit_index < BITS_PER_ELEMENT)
        return array_index * BITS_PER_ELEMENT + bit_index;
    }
    return BITS;
  }

  LIBC_INLINE constexpr void set_bit(uint32_t b) {
    size_t array_index = b / BITS_PER_ELEMENT;
    uint32_t element_index = static_cast<uint32_t>(b % BITS_PER_ELEMENT);
    bit_utils::set_bit(storage[array_index], element_index);
  }

  LIBC_INLINE constexpr void clear_bit(uint32_t b) {
    size_t array_index = b / BITS_PER_ELEMENT;
    uint32_t element_index = static_cast<uint32_t>(b % BITS_PER_ELEMENT);
    bit_utils::clear_bit(storage[array_index], element_index);
  }

  LIBC_INLINE constexpr bool read_bit(uint32_t b) const {
    size_t array_index = b / BITS_PER_ELEMENT;
    uint32_t element_index = static_cast<uint32_t>(b % BITS_PER_ELEMENT);
    return bit_utils::read_bit(storage[array_index], element_index);
  }
};

} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BITFIELD_H
