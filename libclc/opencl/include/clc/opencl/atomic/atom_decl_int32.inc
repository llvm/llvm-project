//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#define __CLC_DECLARE_ATOM(ADDRSPACE, TYPE)                                    \
  _CLC_OVERLOAD _CLC_DECL TYPE __CLC_FUNCTION(volatile ADDRSPACE TYPE *, TYPE);

__CLC_DECLARE_ATOM(__CLC_ADDRESS_SPACE, int)
__CLC_DECLARE_ATOM(__CLC_ADDRESS_SPACE, uint)

#undef __CLC_DECLARE_ATOM

#undef __CLC_FUNCTION
#undef __CLC_ADDRESS_SPACE
