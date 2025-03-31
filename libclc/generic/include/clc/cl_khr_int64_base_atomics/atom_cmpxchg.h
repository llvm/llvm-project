//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL long atom_cmpxchg(volatile global long *p, long cmp, long val);
_CLC_OVERLOAD _CLC_DECL unsigned long atom_cmpxchg(volatile global unsigned long *p, unsigned long cmp, unsigned long val);
_CLC_OVERLOAD _CLC_DECL long atom_cmpxchg(volatile local long *p, long cmp, long val);
_CLC_OVERLOAD _CLC_DECL unsigned long atom_cmpxchg(volatile local unsigned long *p, unsigned long cmp, unsigned long val);
