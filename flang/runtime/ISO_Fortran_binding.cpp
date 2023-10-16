//===-- runtime/ISO_Fortran_binding.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the required interoperability API from ISO_Fortran_binding.h
// as specified in section 18.5.5 of Fortran 2018.

#include "ISO_Fortran_util.h"
#include "terminator.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/type-code.h"
#include <cstdlib>

namespace Fortran::ISO {
extern "C" {

RT_EXT_API_GROUP_BEGIN

RT_API_ATTRS void *CFI_address(
    const CFI_cdesc_t *descriptor, const CFI_index_t subscripts[]) {
  char *p{static_cast<char *>(descriptor->base_addr)};
  const CFI_rank_t rank{descriptor->rank};
  const CFI_dim_t *dim{descriptor->dim};
  for (CFI_rank_t j{0}; j < rank; ++j, ++dim) {
    p += (subscripts[j] - dim->lower_bound) * dim->sm;
  }
  return p;
}

RT_API_ATTRS int CFI_allocate(CFI_cdesc_t *descriptor,
    const CFI_index_t lower_bounds[], const CFI_index_t upper_bounds[],
    std::size_t elem_len) {
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
      descriptor->type > CFI_TYPE_LAST) {
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
    dim->lower_bound = extent == 0 ? 1 : lb;
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

RT_API_ATTRS int CFI_deallocate(CFI_cdesc_t *descriptor) {
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

RT_API_ATTRS int CFI_establish(CFI_cdesc_t *descriptor, void *base_addr,
    CFI_attribute_t attribute, CFI_type_t type, std::size_t elem_len,
    CFI_rank_t rank, const CFI_index_t extents[]) {
  int cfiStatus{VerifyEstablishParameters(descriptor, base_addr, attribute,
      type, elem_len, rank, extents, /*external=*/true)};
  if (cfiStatus != CFI_SUCCESS) {
    return cfiStatus;
  }
  if (type != CFI_type_struct && type != CFI_type_other &&
      !IsCharacterType(type)) {
    elem_len = MinElemLen(type);
  }
  if (elem_len <= 0) {
    return CFI_INVALID_ELEM_LEN;
  }
  EstablishDescriptor(
      descriptor, base_addr, attribute, type, elem_len, rank, extents);
  return CFI_SUCCESS;
}

RT_API_ATTRS int CFI_is_contiguous(const CFI_cdesc_t *descriptor) {
  // See Descriptor::IsContiguous for the rational.
  bool stridesAreContiguous{true};
  CFI_index_t bytes = descriptor->elem_len;
  for (int j{0}; j < descriptor->rank; ++j) {
    stridesAreContiguous &=
        bytes == descriptor->dim[j].sm | descriptor->dim[j].extent == 1;
    bytes *= descriptor->dim[j].extent;
  }
  if (stridesAreContiguous || bytes == 0) {
    return 1;
  }
  return 0;
}

RT_API_ATTRS int CFI_section(CFI_cdesc_t *result, const CFI_cdesc_t *source,
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
  if (runtime::TypeCode{result->type} != runtime::TypeCode{source->type}) {
    return CFI_INVALID_TYPE;
  }
  if (source->elem_len != result->elem_len) {
    return CFI_INVALID_ELEM_LEN;
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
      result->dim[resRank].extent = extent[j];
      result->dim[resRank].lower_bound = extent[j] == 0 ? 1
          : lower_bounds                                ? lower_bounds[j]
                         : source->dim[j].lower_bound;
      result->dim[resRank].sm = actualStride[j] * source->dim[j].sm;
      ++resRank;
    }
  }
  return CFI_SUCCESS;
}

RT_API_ATTRS int CFI_select_part(CFI_cdesc_t *result, const CFI_cdesc_t *source,
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

RT_API_ATTRS int CFI_setpointer(CFI_cdesc_t *result, const CFI_cdesc_t *source,
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
  if (runtime::TypeCode{source->type} != runtime::TypeCode{result->type}) {
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
      CFI_index_t extent{source->dim[j].extent};
      result->dim[j].extent = extent;
      result->dim[j].sm = source->dim[j].sm;
      result->dim[j].lower_bound = extent == 0 ? 1
          : copySrcLB                          ? source->dim[j].lower_bound
                                               : lower_bounds[j];
    }
  }
  return CFI_SUCCESS;
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::ISO
