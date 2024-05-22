//===-- runtime/ISO_Fortran_util.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ISO_FORTRAN_UTIL_H_
#define FORTRAN_RUNTIME_ISO_FORTRAN_UTIL_H_

// Internal utils for establishing CFI_cdesc_t descriptors.

#include "terminator.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/type-code.h"
#include <cstdlib>

namespace Fortran::ISO {
static inline constexpr RT_API_ATTRS bool IsCharacterType(CFI_type_t ty) {
  return ty == CFI_type_char || ty == CFI_type_char16_t ||
      ty == CFI_type_char32_t;
}
static inline constexpr RT_API_ATTRS bool IsAssumedSize(const CFI_cdesc_t *dv) {
  return dv->rank > 0 && dv->dim[dv->rank - 1].extent == -1;
}

static inline RT_API_ATTRS std::size_t MinElemLen(CFI_type_t type) {
  auto typeParams{Fortran::runtime::TypeCode{type}.GetCategoryAndKind()};
  if (!typeParams) {
    Fortran::runtime::Terminator terminator{__FILE__, __LINE__};
    terminator.Crash(
        "not yet implemented: CFI_type_t=%d", static_cast<int>(type));
  }

  return Fortran::runtime::Descriptor::BytesFor(
      typeParams->first, typeParams->second);
}

static inline RT_API_ATTRS int VerifyEstablishParameters(
    CFI_cdesc_t *descriptor, void *base_addr, CFI_attribute_t attribute,
    CFI_type_t type, std::size_t elem_len, CFI_rank_t rank,
    const CFI_index_t extents[], bool external) {
  if (attribute != CFI_attribute_other && attribute != CFI_attribute_pointer &&
      attribute != CFI_attribute_allocatable) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if (rank > CFI_MAX_RANK) {
    return CFI_INVALID_RANK;
  }
  if (base_addr && attribute == CFI_attribute_allocatable) {
    return CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (rank > 0 && base_addr && !extents) {
    return CFI_INVALID_EXTENT;
  }
  if (type < CFI_type_signed_char || type > CFI_TYPE_LAST) {
    return CFI_INVALID_TYPE;
  }
  if (!descriptor) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (external) {
    if (type == CFI_type_struct || type == CFI_type_other ||
        IsCharacterType(type)) {
      if (elem_len <= 0) {
        return CFI_INVALID_ELEM_LEN;
      }
    }
  } else {
    // We do not expect CFI_type_other for internal invocations.
    if (type == CFI_type_other) {
      return CFI_INVALID_TYPE;
    }
  }
  return CFI_SUCCESS;
}

static inline RT_API_ATTRS void EstablishDescriptor(CFI_cdesc_t *descriptor,
    void *base_addr, CFI_attribute_t attribute, CFI_type_t type,
    std::size_t elem_len, CFI_rank_t rank, const CFI_index_t extents[]) {
  descriptor->base_addr = base_addr;
  descriptor->elem_len = elem_len;
  descriptor->version = CFI_VERSION;
  descriptor->rank = rank;
  descriptor->type = type;
  descriptor->attribute = attribute;
  descriptor->f18Addendum = 0;
  std::size_t byteSize{elem_len};
  constexpr std::size_t lower_bound{0};
  if (base_addr) {
    for (std::size_t j{0}; j < rank; ++j) {
      descriptor->dim[j].lower_bound = lower_bound;
      descriptor->dim[j].extent = extents[j];
      descriptor->dim[j].sm = byteSize;
      byteSize *= extents[j];
    }
  }
}
} // namespace Fortran::ISO
#endif // FORTRAN_RUNTIME_ISO_FORTRAN_UTIL_H_
