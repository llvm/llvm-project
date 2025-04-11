//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL int atomic_cmpxchg (volatile local int *, int, int);
_CLC_OVERLOAD _CLC_DECL int atomic_cmpxchg (volatile global int *, int, int);
_CLC_OVERLOAD _CLC_DECL uint atomic_cmpxchg (volatile local uint *, uint, uint);
_CLC_OVERLOAD _CLC_DECL uint atomic_cmpxchg (volatile global uint *, uint, uint);
