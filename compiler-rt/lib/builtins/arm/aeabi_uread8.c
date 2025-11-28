//===-- aeabi_uread8.c - ARM EABI Helper — Unaligned 8-Byte Memory Read----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements __aeabi_uread8 for unaligned memory accesses.
// Reference: Arm RTABI32 Specification.
// https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#unaligned-memory-access
//===-------------------------------------------------------------------------------------===//

typedef struct {
  char v[8];
} v8;

long long __aeabi_uread8(void *p) {
  union {
    v8 v;
    long long u;
  } u;

  u.v = *(v8 *)p;
  return u.u;
}
