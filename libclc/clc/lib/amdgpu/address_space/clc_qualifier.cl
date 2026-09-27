//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/address_space/clc_qualifier.h"

#if _CLC_GENERIC_AS_SUPPORTED

#define __CLC_CONST_TYPE const
#include "clc_qualifier.inc"
#undef __CLC_CONST_TYPE

#define __CLC_CONST_TYPE
#include "clc_qualifier.inc"
#undef __CLC_CONST_TYPE

#endif
