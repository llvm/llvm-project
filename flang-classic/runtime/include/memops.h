/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Various memory operations
 */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(INLINE_MEMOPS)
#ifdef _WIN64
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include "float128.h"

static inline void
__attribute__((always_inline))
__c_mzero1(char *dest, long cnt)
{
  (void) __builtin_memset(dest, 0, (size_t) cnt);
}

static inline void
__attribute__((always_inline))
__c_mzero2(short *dest, long cnt)
{
  (void) __builtin_memset(dest, 0, (size_t) cnt * sizeof(short));
}

static inline void
__attribute__((always_inline))
__c_mzero4(int *dest, long cnt)
{
  (void) __builtin_memset(dest, 0, (size_t) cnt * sizeof(int));
}

static inline void
__attribute__((always_inline))
__c_mzero8(long long *dest, long cnt)
{
  (void) __builtin_memset(dest, 0, (size_t) cnt * sizeof(long long));
}

#ifdef TARGET_SUPPORTS_QUADFP
static inline void __attribute__((always_inline))
__c_mzero16(float128_t *dest, long cnt)
{
  (void)__builtin_memset(dest, 0, (size_t)cnt * sizeof(float128_t));
}
#endif

static inline void
__attribute__((always_inline))
__c_mcopy1(char *dest, char *src, long cnt)
{
  (void) __builtin_memcpy(dest, src, (size_t) cnt);
}

static inline void
__attribute__((always_inline))
__c_mcopy2(short *dest, short *src, long cnt)
{
  (void) __builtin_memcpy(dest, src, (size_t) cnt * sizeof(short));
}

static inline void
__attribute__((always_inline))
__c_mcopy4(int *dest, int *src, long cnt)
{
  (void) __builtin_memcpy(dest, src, (size_t) cnt * sizeof(int));
}

static inline void
__attribute__((always_inline))
__c_mcopy8(long long *dest, long long *src, long cnt)
{
  (void) __builtin_memcpy(dest, src, (size_t) cnt * sizeof(long long));
}

#ifdef TARGET_SUPPORTS_QUADFP
static inline void __attribute__((always_inline))
__c_mcopy16(float128_t *dest, float128_t *src, long cnt)
{
  (void)__builtin_memcpy(dest, src, (size_t)cnt * sizeof(float128_t));
}
#endif

static inline void
__attribute__((always_inline))
__c_mset1(char *dest, int value, long cnt)
{
  ssize_t i;
  for (i = 0; i < cnt; ++i)
    dest[i] = (char) value;
}

static inline void
__attribute__((always_inline))
__c_mset2(short *dest, int value, long cnt)
{
  ssize_t i;
  for (i = 0; i < cnt; ++i)
    dest[i] = (short) value;
}

static inline void
__attribute__((always_inline))
__c_mset4(int *dest, int value, long cnt)
{
  ssize_t i;
  for (i = 0; i < cnt; ++i)
    dest[i] = value;
}

static inline void
__attribute__((always_inline))
__c_mset8(long long *dest, long long value, long cnt)
{
  ssize_t i;
  for (i = 0; i < cnt; ++i)
    dest[i] = (long long) value;
}

#ifdef TARGET_SUPPORTS_QUADFP
static inline void __attribute__((always_inline))
__c_mset16(float128_t *dest, float128_t value, long cnt)
{
  ssize_t i;
  for (i = 0; i < cnt; ++i)
    dest[i] = value;
}
#endif

#else
void __c_mcopy1(char *dest, char *src, long cnt);
void __c_mcopy2(short *dest, short *src, long cnt);
void __c_mcopy4(int *dest, int *src, long cnt);
void __c_mcopy8(long long *dest, long long *src, long cnt);
void __c_mcopy16(float128_t *dest, float128_t *src, long cnt);

void __c_mset1(char *dest, int value, long cnt);
void __c_mset2(short *dest, int value, long cnt);
void __c_mset4(int *dest, int value, long cnt);
void __c_mset8(long long *dest, long long value, long cnt);
void __c_mset16(float128_t *dest, float128_t value, long cnt);

void __c_mzero1(char *dest, long cnt);
void __c_mzero2(short *dest, long cnt);
void __c_mzero4(int *dest, long cnt);
void __c_mzero8(long long *dest, long cnt);
void __c_mzero16(float128_t *dest, long cnt);
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

