//===-- aeabi_uread4.c - ARM EABI Helper — Unaligned 4-Byte Memory Read ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements __aeabi_uread4 for unaligned memory accesses.
// Reference: Arm RTABI32 Specification.
// https://github.com/ARM-software/abi-aa/blob/main/rtabi32/rtabi32.rst#unaligned-memory-access
//===-------------------------------------------------------------------------------------===//

typedef struct {
  char v[4];
} v4;

int __aeabi_uread4(void *p) {
  union {
    v4 v;
    int u;
  } u;

  u.v = *(v4 *)p;
  return u.u;
}
