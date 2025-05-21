//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_FUNCTION remquo

#define __CLC_BODY <clc/math/remquo_decl.inc>
#include <clc/math/gentype.inc>

#if _CLC_GENERIC_AS_SUPPORTED
#define __CLC_BODY <clc/math/remquo_decl.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif

#undef __CLC_FUNCTION
