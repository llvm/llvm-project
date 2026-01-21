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

/* AMO Store Operation Codes (FC values) */
enum _AMO_ST {
  _AMO_ST_ADD = 0x00,  /* Store Add */
  _AMO_ST_XOR = 0x01,  /* Store Xor */
  _AMO_ST_IOR = 0x02,  /* Store Ior */
  _AMO_ST_AND = 0x03,  /* Store And */
  _AMO_ST_UMAX = 0x04, /* Store Unsigned Maximum */
  _AMO_ST_SMAX = 0x05, /* Store Signed Maximum */
  _AMO_ST_UMIN = 0x06, /* Store Unsigned Minimum */
  _AMO_ST_SMIN = 0x07, /* Store Signed Minimum */
  _AMO_ST_TWIN = 0x18  /* Store Twin */
};

/* 32-bit unsigned AMO store operations */
static inline void amo_stwat_add(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_ADD);
}

static inline void amo_stwat_xor(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_XOR);
}

static inline void amo_stwat_ior(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_IOR);
}

static inline void amo_stwat_and(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_AND);
}

static inline void amo_stwat_umax(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_UMAX);
}

static inline void amo_stwat_umin(uint32_t *ptr, uint32_t val) {
  __builtin_amo_stwat(ptr, val, _AMO_ST_UMIN);
}

/* 32-bit signed AMO store operations */
static inline void amo_stwat_sadd(int32_t *ptr, int32_t val) {
  __builtin_amo_stwat_s(ptr, val, _AMO_ST_ADD);
}

static inline void amo_stwat_smax(int32_t *ptr, int32_t val) {
  __builtin_amo_stwat_s(ptr, val, _AMO_ST_SMAX);
}

static inline void amo_stwat_smin(int32_t *ptr, int32_t val) {
  __builtin_amo_stwat_s(ptr, val, _AMO_ST_SMIN);
}

/* 64-bit unsigned AMO store operations */
static inline void amo_stdat_add(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_ADD);
}

static inline void amo_stdat_xor(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_XOR);
}

static inline void amo_stdat_ior(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_IOR);
}

static inline void amo_stdat_and(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_AND);
}

static inline void amo_stdat_umax(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_UMAX);
}

static inline void amo_stdat_umin(uint64_t *ptr, uint64_t val) {
  __builtin_amo_stdat(ptr, val, _AMO_ST_UMIN);
}

/* 64-bit signed AMO store operations */
static inline void amo_stdat_sadd(int64_t *ptr, int64_t val) {
  __builtin_amo_stdat_s(ptr, val, _AMO_ST_ADD);
}

static inline void amo_stdat_smax(int64_t *ptr, int64_t val) {
  __builtin_amo_stdat_s(ptr, val, _AMO_ST_SMAX);
}

static inline void amo_stdat_smin(int64_t *ptr, int64_t val) {
  __builtin_amo_stdat_s(ptr, val, _AMO_ST_SMIN);
}

#ifdef __cplusplus
}
#endif

#endif /* _AMO_H */
