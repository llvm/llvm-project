/*===-- include/flang/ISO_Fortran_binding.h -----------------------*- C++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef CFI_ISO_FORTRAN_BINDING_H_
#define CFI_ISO_FORTRAN_BINDING_H_

#include <stddef.h>

/* Standard interface to Fortran from C and C++.
 * These interfaces are named in subclause 18.5 of the Fortran 2018
 * standard, with most of the actual details being left to the
 * implementation.
 */

#ifdef __cplusplus
namespace Fortran {
namespace ISO {
inline namespace Fortran_2018 {
#endif

/* 18.5.4 */
#define CFI_VERSION 20180515

#define CFI_MAX_RANK 15
typedef unsigned char CFI_rank_t;

/* This type is probably larger than a default Fortran INTEGER
 * and should be used for all array indexing and loop bound calculations.
 */
typedef ptrdiff_t CFI_index_t;

typedef unsigned char CFI_attribute_t;
#define CFI_attribute_pointer 1
#define CFI_attribute_allocatable 2
#define CFI_attribute_other 0 /* neither pointer nor allocatable */

typedef signed char CFI_type_t;
/* These codes are required to be macros (i.e., #ifdef will work).
 * They are not required to be distinct, but neither are they required
 * to have had their synonyms combined.  Codes marked as extensions may be
 * place holders for as yet unimplemented types.
 */
#define CFI_type_signed_char 1
#define CFI_type_short 2
#define CFI_type_int 3
#define CFI_type_long 4
#define CFI_type_long_long 5
#define CFI_type_size_t 6
#define CFI_type_int8_t 7
#define CFI_type_int16_t 8
#define CFI_type_int32_t 9
#define CFI_type_int64_t 10
#define CFI_type_int128_t 11 /* extension */
#define CFI_type_int_least8_t 12
#define CFI_type_int_least16_t 13
#define CFI_type_int_least32_t 14
#define CFI_type_int_least64_t 15
#define CFI_type_int_least128_t 16 /* extension */
#define CFI_type_int_fast8_t 17
#define CFI_type_int_fast16_t 18
#define CFI_type_int_fast32_t 19
#define CFI_type_int_fast64_t 20
#define CFI_type_int_fast128_t 21 /* extension */
#define CFI_type_intmax_t 22
#define CFI_type_intptr_t 23
#define CFI_type_ptrdiff_t 24
#define CFI_type_half_float 25 /* extension: kind=2 */
#define CFI_type_bfloat 26 /* extension: kind=3 */
#define CFI_type_float 27
#define CFI_type_double 28
#define CFI_type_extended_double 29 /* extension: kind=10 */
#define CFI_type_long_double 30
#define CFI_type_float128 31 /* extension: kind=16 */
#define CFI_type_half_float_Complex 32 /* extension: kind=2 */
#define CFI_type_bfloat_Complex 33 /* extension: kind=3 */
#define CFI_type_float_Complex 34
#define CFI_type_double_Complex 35
#define CFI_type_extended_double_Complex 36 /* extension: kind=10 */
#define CFI_type_long_double_Complex 37
#define CFI_type_float128_Complex 38 /* extension: kind=16 */
#define CFI_type_Bool 39
#define CFI_type_char 40
#define CFI_type_cptr 41
#define CFI_type_struct 42
#define CFI_type_char16_t 43 /* extension */
#define CFI_type_char32_t 44 /* extension */
#define CFI_TYPE_LAST CFI_type_char32_t
#define CFI_type_other (-1) // must be negative

/* Error code macros - skip some of the small values to avoid conflicts with
 * other status codes mandated by the standard, e.g. those returned by
 * GET_ENVIRONMENT_VARIABLE (16.9.84) */
#define CFI_SUCCESS 0 /* must be zero */
#define CFI_ERROR_BASE_ADDR_NULL 11
#define CFI_ERROR_BASE_ADDR_NOT_NULL 12
#define CFI_INVALID_ELEM_LEN 13
#define CFI_INVALID_RANK 14
#define CFI_INVALID_TYPE 15
#define CFI_INVALID_ATTRIBUTE 16
#define CFI_INVALID_EXTENT 17
#define CFI_INVALID_DESCRIPTOR 18
#define CFI_ERROR_MEM_ALLOCATION 19
#define CFI_ERROR_OUT_OF_BOUNDS 20

/* 18.5.2 per-dimension information */
typedef struct CFI_dim_t {
  CFI_index_t lower_bound;
  CFI_index_t extent; /* == -1 for assumed size */
  CFI_index_t sm; /* memory stride in bytes */
} CFI_dim_t;

#ifdef __cplusplus
namespace cfi_internal {
// C++ does not support flexible array.
// The below structure emulates a flexible array. This structure does not take
// care of getting the memory storage. Note that it already contains one element
// because a struct cannot be empty.
template <typename T> struct FlexibleArray : T {
  T &operator[](int index) { return *(this + index); }
  const T &operator[](int index) const { return *(this + index); }
  operator T *() { return this; }
  operator const T *() const { return this; }
};
} // namespace cfi_internal
#endif

/* 18.5.3 generic data descriptor */
typedef struct CFI_cdesc_t {
  /* These three members must appear first, in exactly this order. */
  void *base_addr;
  size_t elem_len; /* element size in bytes */
  int version; /* == CFI_VERSION */
  CFI_rank_t rank; /* [0 .. CFI_MAX_RANK] */
  CFI_type_t type;
  CFI_attribute_t attribute;
  unsigned char f18Addendum;
#ifdef __cplusplus
  cfi_internal::FlexibleArray<CFI_dim_t> dim;
#else
  CFI_dim_t dim[]; /* must appear last */
#endif
} CFI_cdesc_t;

/* 18.5.4 */
#ifdef __cplusplus
// The struct below take care of getting the memory storage for C++ CFI_cdesc_t
// that contain an emulated flexible array.
namespace cfi_internal {
template <int r> struct CdescStorage : public CFI_cdesc_t {
  static_assert((r > 1 && r <= CFI_MAX_RANK), "CFI_INVALID_RANK");
  CFI_dim_t dim[r - 1];
};
template <> struct CdescStorage<1> : public CFI_cdesc_t {};
template <> struct CdescStorage<0> : public CFI_cdesc_t {};
} // namespace cfi_internal
#define CFI_CDESC_T(rank) cfi_internal::CdescStorage<rank>
#else
#define CFI_CDESC_T(rank) \
  struct { \
    CFI_cdesc_t cdesc; /* must be first */ \
    CFI_dim_t dim[rank]; \
  }
#endif

/* 18.5.5 procedural interfaces*/
#ifdef __cplusplus
extern "C" {
#endif
void *CFI_address(const CFI_cdesc_t *, const CFI_index_t subscripts[]);
int CFI_allocate(CFI_cdesc_t *, const CFI_index_t lower_bounds[],
    const CFI_index_t upper_bounds[], size_t elem_len);
int CFI_deallocate(CFI_cdesc_t *);
int CFI_establish(CFI_cdesc_t *, void *base_addr, CFI_attribute_t, CFI_type_t,
    size_t elem_len, CFI_rank_t, const CFI_index_t extents[]);
int CFI_is_contiguous(const CFI_cdesc_t *);
int CFI_section(CFI_cdesc_t *, const CFI_cdesc_t *source,
    const CFI_index_t lower_bounds[], const CFI_index_t upper_bounds[],
    const CFI_index_t strides[]);
int CFI_select_part(CFI_cdesc_t *, const CFI_cdesc_t *source,
    size_t displacement, size_t elem_len);
int CFI_setpointer(
    CFI_cdesc_t *, const CFI_cdesc_t *source, const CFI_index_t lower_bounds[]);
#ifdef __cplusplus
} // extern "C"
} // inline namespace Fortran_2018
}
}
#endif

#endif /* CFI_ISO_FORTRAN_BINDING_H_ */
