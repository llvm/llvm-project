//===-- runtime/ISO_Fortran_binding.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the required interoperability API from ISO_Fortran_binding.h
// as specified in section 18.5.5 of Fortran 2018.

#include "../include/flang/ISO_Fortran_binding.h"
#include "descriptor.h"
#include <cstdlib>

namespace Fortran::ISO {
extern "C" {

static inline constexpr bool IsCharacterType(CFI_type_t ty) {
  return ty == CFI_type_char || ty == CFI_type_char16_t ||
      ty == CFI_type_char32_t;
}
static inline constexpr bool IsAssumedSize(const CFI_cdesc_t *dv) {
  return dv->rank > 0 && dv->dim[dv->rank - 1].extent == -1;
}

void *CFI_address(
    const CFI_cdesc_t *descriptor, const CFI_index_t subscripts[]) {
  char *p{static_cast<char *>(descriptor->base_addr)};
  const CFI_rank_t rank{descriptor->rank};
  const CFI_dim_t *dim{descriptor->dim};
  for (CFI_rank_t j{0}; j < rank; ++j, ++dim) {
    p += (subscripts[j] - dim->lower_bound) * dim->sm;
  }
  return p;
}

int CFI_allocate(CFI_cdesc_t *descriptor, const CFI_index_t lower_bounds[],
    const CFI_index_t upper_bounds[], std::size_t elem_len) {
  if (!descriptor) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->attribute != CFI_attribute_allocatable &&
      descriptor->attribute != CFI_attribute_pointer) {
    // Non-interoperable object
    return CFI_INVALID_ATTRIBUTE;
  }
  if (descriptor->attribute == CFI_attribute_allocatable &&
      descriptor->base_addr) {
    return CFI_ERROR_BASE_ADDR_NOT_NULL;
  }
  if (descriptor->rank > CFI_MAX_RANK) {
    return CFI_INVALID_RANK;
  }
  if (descriptor->type < CFI_type_signed_char ||
      descriptor->type > CFI_type_struct) {
    return CFI_INVALID_TYPE;
  }
  if (!IsCharacterType(descriptor->type)) {
    elem_len = descriptor->elem_len;
    if (elem_len <= 0) {
      return CFI_INVALID_ELEM_LEN;
    }
  }
  std::size_t rank{descriptor->rank};
  CFI_dim_t *dim{descriptor->dim};
  std::size_t byteSize{elem_len};
  for (std::size_t j{0}; j < rank; ++j, ++dim) {
    CFI_index_t lb{lower_bounds[j]};
    CFI_index_t ub{upper_bounds[j]};
    CFI_index_t extent{ub >= lb ? ub - lb + 1 : 0};
    dim->lower_bound = lb;
    dim->extent = extent;
    dim->sm = byteSize;
    byteSize *= extent;
  }
  void *p{std::malloc(byteSize)};
  if (!p && byteSize) {
    return CFI_ERROR_MEM_ALLOCATION;
  }
  descriptor->base_addr = p;
  descriptor->elem_len = elem_len;
  return CFI_SUCCESS;
}

int CFI_deallocate(CFI_cdesc_t *descriptor) {
  if (!descriptor) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->attribute != CFI_attribute_allocatable &&
      descriptor->attribute != CFI_attribute_pointer) {
    // Non-interoperable object
    return CFI_INVALID_DESCRIPTOR;
  }
  if (!descriptor->base_addr) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }
  std::free(descriptor->base_addr);
  descriptor->base_addr = nullptr;
  return CFI_SUCCESS;
}

static constexpr std::size_t MinElemLen(CFI_type_t type) {
  std::size_t minElemLen{0};
  switch (type) {
  case CFI_type_signed_char:
    minElemLen = sizeof(signed char);
    break;
  case CFI_type_short:
    minElemLen = sizeof(short);
    break;
  case CFI_type_int:
    minElemLen = sizeof(int);
    break;
  case CFI_type_long:
    minElemLen = sizeof(long);
    break;
  case CFI_type_long_long:
    minElemLen = sizeof(long long);
    break;
  case CFI_type_size_t:
    minElemLen = sizeof(std::size_t);
    break;
  case CFI_type_int8_t:
    minElemLen = sizeof(std::int8_t);
    break;
  case CFI_type_int16_t:
    minElemLen = sizeof(std::int16_t);
    break;
  case CFI_type_int32_t:
    minElemLen = sizeof(std::int32_t);
    break;
  case CFI_type_int64_t:
    minElemLen = sizeof(std::int64_t);
    break;
  case CFI_type_int128_t:
    minElemLen = 2 * sizeof(std::int64_t);
    break;
  case CFI_type_int_least8_t:
    minElemLen = sizeof(std::int_least8_t);
    break;
  case CFI_type_int_least16_t:
    minElemLen = sizeof(std::int_least16_t);
    break;
  case CFI_type_int_least32_t:
    minElemLen = sizeof(std::int_least32_t);
    break;
  case CFI_type_int_least64_t:
    minElemLen = sizeof(std::int_least64_t);
    break;
  case CFI_type_int_least128_t:
    minElemLen = 2 * sizeof(std::int_least64_t);
    break;
  case CFI_type_int_fast8_t:
    minElemLen = sizeof(std::int_fast8_t);
    break;
  case CFI_type_int_fast16_t:
    minElemLen = sizeof(std::int_fast16_t);
    break;
  case CFI_type_int_fast32_t:
    minElemLen = sizeof(std::int_fast32_t);
    break;
  case CFI_type_int_fast64_t:
    minElemLen = sizeof(std::int_fast64_t);
    break;
  case CFI_type_intmax_t:
    minElemLen = sizeof(std::intmax_t);
    break;
  case CFI_type_intptr_t:
    minElemLen = sizeof(std::intptr_t);
    break;
  case CFI_type_ptrdiff_t:
    minElemLen = sizeof(std::ptrdiff_t);
    break;
  case CFI_type_float:
    minElemLen = sizeof(float);
    break;
  case CFI_type_double:
    minElemLen = sizeof(double);
    break;
  case CFI_type_long_double:
    minElemLen = sizeof(long double);
    break;
  case CFI_type_float_Complex:
    minElemLen = 2 * sizeof(float);
    break;
  case CFI_type_double_Complex:
    minElemLen = 2 * sizeof(double);
    break;
  case CFI_type_long_double_Complex:
    minElemLen = 2 * sizeof(long double);
    break;
  case CFI_type_Bool:
    minElemLen = 1;
    break;
  case CFI_type_cptr:
    minElemLen = sizeof(void *);
    break;
  case CFI_type_char16_t:
    minElemLen = sizeof(char16_t);
    break;
  case CFI_type_char32_t:
    minElemLen = sizeof(char32_t);
    break;
  }
  return minElemLen;
}

int CFI_establish(CFI_cdesc_t *descriptor, void *base_addr,
    CFI_attribute_t attribute, CFI_type_t type, std::size_t elem_len,
    CFI_rank_t rank, const CFI_index_t extents[]) {
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
  if (type < CFI_type_signed_char || type > CFI_type_struct) {
    return CFI_INVALID_TYPE;
  }
  if (!descriptor) {
    return CFI_INVALID_DESCRIPTOR;
  }
  std::size_t minElemLen{MinElemLen(type)};
  if (minElemLen > 0) {
    elem_len = minElemLen;
  } else if (elem_len <= 0) {
    return CFI_INVALID_ELEM_LEN;
  }
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
  return CFI_SUCCESS;
}

int CFI_is_contiguous(const CFI_cdesc_t *descriptor) {
  CFI_index_t bytes = descriptor->elem_len;
  for (int j{0}; j < descriptor->rank; ++j) {
    if (bytes != descriptor->dim[j].sm) {
      return 0;
    }
    bytes *= descriptor->dim[j].extent;
  }
  return 1;
}

int CFI_section(CFI_cdesc_t *result, const CFI_cdesc_t *source,
    const CFI_index_t lower_bounds[], const CFI_index_t upper_bounds[],
    const CFI_index_t strides[]) {
  CFI_index_t extent[CFI_MAX_RANK];
  CFI_index_t actualStride[CFI_MAX_RANK];
  CFI_rank_t resRank{0};

  if (!result || !source) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (source->rank == 0) {
    return CFI_INVALID_RANK;
  }
  if (IsAssumedSize(source) && !upper_bounds) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if ((result->type != source->type) ||
      (result->elem_len != source->elem_len)) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (result->attribute == CFI_attribute_allocatable) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if (!source->base_addr) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }

  char *shiftedBaseAddr{static_cast<char *>(source->base_addr)};
  bool isZeroSized{false};
  for (int j{0}; j < source->rank; ++j) {
    const CFI_dim_t &dim{source->dim[j]};
    const CFI_index_t srcLB{dim.lower_bound};
    const CFI_index_t srcUB{srcLB + dim.extent - 1};
    const CFI_index_t lb{lower_bounds ? lower_bounds[j] : srcLB};
    const CFI_index_t ub{upper_bounds ? upper_bounds[j] : srcUB};
    const CFI_index_t stride{strides ? strides[j] : 1};

    if (stride == 0 && lb != ub) {
      return CFI_ERROR_OUT_OF_BOUNDS;
    }
    if ((lb <= ub && stride >= 0) || (lb >= ub && stride < 0)) {
      if ((lb < srcLB) || (lb > srcUB) || (ub < srcLB) || (ub > srcUB)) {
        return CFI_ERROR_OUT_OF_BOUNDS;
      }
      shiftedBaseAddr += (lb - srcLB) * dim.sm;
      extent[j] = stride != 0 ? 1 + (ub - lb) / stride : 1;
    } else {
      isZeroSized = true;
      extent[j] = 0;
    }
    actualStride[j] = stride;
    resRank += (stride != 0);
  }
  if (resRank != result->rank) {
    return CFI_INVALID_DESCRIPTOR;
  }

  // For zero-sized arrays, base_addr is processor-dependent (see 18.5.3).
  // We keep it on the source base_addr
  result->base_addr = isZeroSized ? source->base_addr : shiftedBaseAddr;
  resRank = 0;
  for (int j{0}; j < source->rank; ++j) {
    if (actualStride[j] != 0) {
      result->dim[resRank].lower_bound = 0;
      result->dim[resRank].extent = extent[j];
      result->dim[resRank].sm = actualStride[j] * source->dim[j].sm;
      ++resRank;
    }
  }
  return CFI_SUCCESS;
}

int CFI_select_part(CFI_cdesc_t *result, const CFI_cdesc_t *source,
    std::size_t displacement, std::size_t elem_len) {
  if (!result || !source) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (result->rank != source->rank) {
    return CFI_INVALID_RANK;
  }
  if (result->attribute == CFI_attribute_allocatable) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if (!source->base_addr) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }
  if (IsAssumedSize(source)) {
    return CFI_INVALID_DESCRIPTOR;
  }

  if (!IsCharacterType(result->type)) {
    elem_len = result->elem_len;
  }
  if (displacement + elem_len > source->elem_len) {
    return CFI_INVALID_ELEM_LEN;
  }

  result->base_addr = displacement + static_cast<char *>(source->base_addr);
  result->elem_len = elem_len;
  for (int j{0}; j < source->rank; ++j) {
    result->dim[j].lower_bound = 0;
    result->dim[j].extent = source->dim[j].extent;
    result->dim[j].sm = source->dim[j].sm;
  }
  return CFI_SUCCESS;
}

int CFI_setpointer(CFI_cdesc_t *result, const CFI_cdesc_t *source,
    const CFI_index_t lower_bounds[]) {
  if (!result) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (result->attribute != CFI_attribute_pointer) {
    return CFI_INVALID_ATTRIBUTE;
  }
  if (!source) {
    result->base_addr = nullptr;
    return CFI_SUCCESS;
  }
  if (source->rank != result->rank) {
    return CFI_INVALID_RANK;
  }
  if (source->type != result->type) {
    return CFI_INVALID_TYPE;
  }
  if (source->elem_len != result->elem_len) {
    return CFI_INVALID_ELEM_LEN;
  }
  if (!source->base_addr && source->attribute != CFI_attribute_pointer) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }
  if (IsAssumedSize(source)) {
    return CFI_INVALID_DESCRIPTOR;
  }

  const bool copySrcLB{!lower_bounds};
  result->base_addr = source->base_addr;
  if (source->base_addr) {
    for (int j{0}; j < result->rank; ++j) {
      result->dim[j].extent = source->dim[j].extent;
      result->dim[j].sm = source->dim[j].sm;
      result->dim[j].lower_bound =
          copySrcLB ? source->dim[j].lower_bound : lower_bounds[j];
    }
  }
  return CFI_SUCCESS;
}
} // extern "C"
} // namespace Fortran::ISO
