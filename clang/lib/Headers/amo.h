/*===---- amo.h - PowerPC Atomic Memory Operations ------------------------===*\
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
\*===----------------------------------------------------------------------===*/

/* This header provides compatibility for GCC's AMO functions.
 * The functions here call Clang's underlying AMO builtins.
 */

#ifndef _AMO_H
#define _AMO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* AMO Load Operation Codes (FC values) */
enum {
  _AMO_LD_ADD = 0x00,  /* Fetch and Add */
  _AMO_LD_XOR = 0x01,  /* Fetch and XOR */
  _AMO_LD_IOR = 0x02,  /* Fetch and OR */
  _AMO_LD_AND = 0x03,  /* Fetch and AND */
  _AMO_LD_UMAX = 0x04, /* Fetch and Maximum Unsigned */
  _AMO_LD_SMAX = 0x05, /* Fetch and Maximum Signed */
  _AMO_LD_UMIN = 0x06, /* Fetch and Minimum Unsigned */
  _AMO_LD_SMIN = 0x07, /* Fetch and Minimum Signed */
  _AMO_LD_SWAP = 0x08  /* Swap */
};

/* 32-bit unsigned AMO load operations */
static inline uint32_t amo_lwat_add(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_ADD);
}

static inline uint32_t amo_lwat_xor(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_XOR);
}

static inline uint32_t amo_lwat_ior(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_IOR);
}

static inline uint32_t amo_lwat_and(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_AND);
}

static inline uint32_t amo_lwat_umax(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_UMAX);
}

static inline uint32_t amo_lwat_umin(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_UMIN);
}

static inline uint32_t amo_lwat_swap(uint32_t *ptr, uint32_t val) {
  return __builtin_amo_lwat(ptr, val, _AMO_LD_SWAP);
}

/* 32-bit signed AMO load operations */
static inline int32_t amo_lwat_sadd(int32_t *ptr, int32_t val) {
  return __builtin_amo_lwat_s(ptr, val, _AMO_LD_ADD);
}

static inline int32_t amo_lwat_smax(int32_t *ptr, int32_t val) {
  return __builtin_amo_lwat_s(ptr, val, _AMO_LD_SMAX);
}

static inline int32_t amo_lwat_smin(int32_t *ptr, int32_t val) {
  return __builtin_amo_lwat_s(ptr, val, _AMO_LD_SMIN);
}

static inline int32_t amo_lwat_sswap(int32_t *ptr, int32_t val) {
  return __builtin_amo_lwat_s(ptr, val, _AMO_LD_SWAP);
}

/* 64-bit unsigned AMO load operations */
static inline uint64_t amo_ldat_add(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_ADD);
}

static inline uint64_t amo_ldat_xor(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_XOR);
}

static inline uint64_t amo_ldat_ior(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_IOR);
}

static inline uint64_t amo_ldat_and(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_AND);
}

static inline uint64_t amo_ldat_umax(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_UMAX);
}

static inline uint64_t amo_ldat_umin(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_UMIN);
}

static inline uint64_t amo_ldat_swap(uint64_t *ptr, uint64_t val) {
  return __builtin_amo_ldat(ptr, val, _AMO_LD_SWAP);
}

/* 64-bit signed AMO load operations */
static inline int64_t amo_ldat_sadd(int64_t *ptr, int64_t val) {
  return __builtin_amo_ldat_s(ptr, val, _AMO_LD_ADD);
}

static inline int64_t amo_ldat_smax(int64_t *ptr, int64_t val) {
  return __builtin_amo_ldat_s(ptr, val, _AMO_LD_SMAX);
}

static inline int64_t amo_ldat_smin(int64_t *ptr, int64_t val) {
  return __builtin_amo_ldat_s(ptr, val, _AMO_LD_SMIN);
}

static inline int64_t amo_ldat_sswap(int64_t *ptr, int64_t val) {
  return __builtin_amo_ldat_s(ptr, val, _AMO_LD_SWAP);
}

#ifdef __cplusplus
}
#endif

#endif /* _AMO_H */
