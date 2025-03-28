//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL int atom_cmpxchg(volatile local int *p, int cmp, int val);
_CLC_OVERLOAD _CLC_DECL unsigned int atom_cmpxchg(volatile local unsigned int *p, unsigned int cmp, unsigned int val);
