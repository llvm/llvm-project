//===-- include/flang/Runtime/descriptor-consts.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_DESCRIPTOR_CONSTS_H_
#define FORTRAN_RUNTIME_DESCRIPTOR_CONSTS_H_

#include "flang/Common/api-attrs.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include <cstddef>
#include <cstdint>

// Value of the addendum presence flag.
#define _CFI_ADDENDUM_FLAG 1
// Number of bits needed to be shifted when manipulating the allocator index.
#define _CFI_ALLOCATOR_IDX_SHIFT 1
// Allocator index mask.
#define _CFI_ALLOCATOR_IDX_MASK 0b00001110

namespace Fortran::runtime::typeInfo {
using TypeParameterValue = std::int64_t;
class DerivedType;
} // namespace Fortran::runtime::typeInfo

namespace Fortran::runtime {
class Descriptor;
using SubscriptValue = ISO::CFI_index_t;

/// Returns size in bytes of the descriptor (not the data)
/// This must be at least as large as the largest descriptor of any target
/// triple.
static constexpr RT_API_ATTRS std::size_t MaxDescriptorSizeInBytes(
    int rank, bool addendum = false, int lengthTypeParameters = 0) {
  // Layout:
  //
  // fortran::runtime::Descriptor {
  //   ISO::CFI_cdesc_t {
  //     void *base_addr;           (pointer -> up to 8 bytes)
  //     size_t elem_len;           (up to 8 bytes)
  //     int version;               (up to 4 bytes)
  //     CFI_rank_t rank;           (unsigned char -> 1 byte)
  //     CFI_type_t type;           (signed char -> 1 byte)
  //     CFI_attribute_t attribute; (unsigned char -> 1 byte)
  //     unsigned char extra;       (1 byte)
  //   }
  // }
  // fortran::runtime::Dimension[rank] {
  //   ISO::CFI_dim_t {
  //     CFI_index_t lower_bound; (ptrdiff_t -> up to 8 bytes)
  //     CFI_index_t extent;      (ptrdiff_t -> up to 8 bytes)
  //     CFI_index_t sm;          (ptrdiff_t -> up to 8 bytes)
  //   }
  // }
  // fortran::runtime::DescriptorAddendum {
  //   const typeInfo::DerivedType *derivedType_;        (pointer -> up to 8
  //   bytes) typeInfo::TypeParameterValue len_[lenParameters]; (int64_t -> 8
  //   bytes)
  // }
  std::size_t bytes{24u + rank * 24u};
  if (addendum || lengthTypeParameters > 0) {
    if (lengthTypeParameters < 1)
      lengthTypeParameters = 1;
    bytes += 8u + static_cast<std::size_t>(lengthTypeParameters) * 8u;
  }
  return bytes;
}

} // namespace Fortran::runtime

#endif /* FORTRAN_RUNTIME_DESCRIPTOR_CONSTS_H_ */
