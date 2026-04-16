/*===---- arm64intr.h - ARM64 Windows intrinsics -------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* Only include this if we're compiling for the windows platform. */
#ifndef _MSC_VER
#include_next <arm64intr.h>
#else

#ifndef __ARM64INTR_H
#define __ARM64INTR_H

/* Encode an AArch64 system register for use with
   _ReadStatusReg/_WriteStatusReg. op0 must be 2 or 3; only the low bit is
   stored. */
#define ARM64_SYSREG(op0, op1, CRn, CRm, op2)                                  \
  ((((op0) & 0x1) << 14) | (((op1) & 0x7) << 11) | (((CRn) & 0xF) << 7) |      \
   (((CRm) & 0xF) << 3) | ((op2) & 0x7))

#define ARM64_FPCR ARM64_SYSREG(3, 3, 4, 4, 0)
#define ARM64_FPSR ARM64_SYSREG(3, 3, 4, 4, 1)

typedef enum
{
  _ARM64_BARRIER_SY    = 0xF,
  _ARM64_BARRIER_ST    = 0xE,
  _ARM64_BARRIER_LD    = 0xD,
  _ARM64_BARRIER_ISH   = 0xB,
  _ARM64_BARRIER_ISHST = 0xA,
  _ARM64_BARRIER_ISHLD = 0x9,
  _ARM64_BARRIER_NSH   = 0x7,
  _ARM64_BARRIER_NSHST = 0x6,
  _ARM64_BARRIER_NSHLD = 0x5,
  _ARM64_BARRIER_OSH   = 0x3,
  _ARM64_BARRIER_OSHST = 0x2,
  _ARM64_BARRIER_OSHLD = 0x1
} _ARM64INTR_BARRIER_TYPE;

#ifdef __cplusplus
extern "C" {
#endif

unsigned __int8 __ldar8(const volatile unsigned __int8 *);
unsigned __int16 __ldar16(const volatile unsigned __int16 *);
unsigned __int32 __ldar32(const volatile unsigned __int32 *);
unsigned __int64 __ldar64(const volatile unsigned __int64 *);

void __stlr8(unsigned __int8 volatile *, unsigned __int8);
void __stlr16(unsigned __int16 volatile *, unsigned __int16);
void __stlr32(unsigned __int32 volatile *, unsigned __int32);
void __stlr64(unsigned __int64 volatile *, unsigned __int64);

unsigned __int8 __ldxr8(const volatile unsigned __int8 *);
unsigned __int16 __ldxr16(const volatile unsigned __int16 *);
unsigned __int32 __ldxr32(const volatile unsigned __int32 *);
unsigned __int64 __ldxr64(const volatile unsigned __int64 *);

unsigned __int8 __ldaxr8(const volatile unsigned __int8 *);
unsigned __int16 __ldaxr16(const volatile unsigned __int16 *);
unsigned __int32 __ldaxr32(const volatile unsigned __int32 *);
unsigned __int64 __ldaxr64(const volatile unsigned __int64 *);

unsigned __int8 __stxr8(volatile unsigned __int8 *, unsigned __int8);
unsigned __int8 __stxr16(volatile unsigned __int16 *, unsigned __int16);
unsigned __int8 __stxr32(volatile unsigned __int32 *, unsigned __int32);
unsigned __int8 __stxr64(volatile unsigned __int64 *, unsigned __int64);

unsigned __int8 __stlxr8(volatile unsigned __int8 *, unsigned __int8);
unsigned __int8 __stlxr16(volatile unsigned __int16 *, unsigned __int16);
unsigned __int8 __stlxr32(volatile unsigned __int32 *, unsigned __int32);
unsigned __int8 __stlxr64(volatile unsigned __int64 *, unsigned __int64);

void __clrex(unsigned __int8);

unsigned __int8 __cas8(unsigned __int8 volatile *, unsigned __int8,
                       unsigned __int8);
unsigned __int16 __cas16(unsigned __int16 volatile *, unsigned __int16,
                         unsigned __int16);
unsigned __int32 __cas32(unsigned __int32 volatile *, unsigned __int32,
                         unsigned __int32);
unsigned __int64 __cas64(unsigned __int64 volatile *, unsigned __int64,
                         unsigned __int64);

unsigned __int8 __casa8(unsigned __int8 volatile *, unsigned __int8,
                        unsigned __int8);
unsigned __int16 __casa16(unsigned __int16 volatile *, unsigned __int16,
                          unsigned __int16);
unsigned __int32 __casa32(unsigned __int32 volatile *, unsigned __int32,
                          unsigned __int32);
unsigned __int64 __casa64(unsigned __int64 volatile *, unsigned __int64,
                          unsigned __int64);

unsigned __int8 __casl8(unsigned __int8 volatile *, unsigned __int8,
                        unsigned __int8);
unsigned __int16 __casl16(unsigned __int16 volatile *, unsigned __int16,
                          unsigned __int16);
unsigned __int32 __casl32(unsigned __int32 volatile *, unsigned __int32,
                          unsigned __int32);
unsigned __int64 __casl64(unsigned __int64 volatile *, unsigned __int64,
                          unsigned __int64);

unsigned __int8 __casal8(unsigned __int8 volatile *, unsigned __int8,
                         unsigned __int8);
unsigned __int16 __casal16(unsigned __int16 volatile *, unsigned __int16,
                           unsigned __int16);
unsigned __int32 __casal32(unsigned __int32 volatile *, unsigned __int32,
                           unsigned __int32);
unsigned __int64 __casal64(unsigned __int64 volatile *, unsigned __int64,
                           unsigned __int64);

#ifdef __cplusplus
}
#endif

#endif /* __ARM64INTR_H */
#endif /* _MSC_VER */
