//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_CLC_REMQUO_H__
#define __CLC_MATH_CLC_REMQUO_H__

#define __CLC_FUNCTION __clc_remquo

#define __CLC_BODY <clc/math/remquo_decl.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>

#undef __CLC_ADDRESS_SPACE
#undef __CLC_FUNCTION

#endif // __CLC_MATH_CLC_REMQUO_H__
