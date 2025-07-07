//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_ATOMIC_CLC_ATOMIC_FETCH_OR_H__
#define __CLC_ATOMIC_CLC_ATOMIC_FETCH_OR_H__

#include <clc/internal/clc.h>

#define FUNCTION __clc_atomic_fetch_or

#define __CLC_BODY <clc/atomic/atomic_decl.inc>
#include <clc/integer/gentype.inc>

#undef FUNCTION

#endif // __CLC_ATOMIC_CLC_ATOMIC_FETCH_OR_H__
