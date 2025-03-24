//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_FUNCTION atomic_xchg

_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION (volatile local float *, float);
_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION (volatile global float *, float);
#include <clc/atomic/atomic_decl.inc>
