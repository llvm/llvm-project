//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_RELATIONAL_CLC_SELECT_H__
#define __CLC_RELATIONAL_CLC_SELECT_H__

#include <clc/utils.h>

#define __CLC_SELECT_FN __clc_select

#define __CLC_BODY <clc/relational/clc_select_decl.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select_decl.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_SELECT_FN

#endif // __CLC_RELATIONAL_CLC_SELECT_H__
