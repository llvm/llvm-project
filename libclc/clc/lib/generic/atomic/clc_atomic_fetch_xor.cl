//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_xor.h>

#define FUNCTION __clc_atomic_fetch_xor
#define __IMPL_FUNCTION __scoped_atomic_fetch_xor

#define __CLC_BODY <clc_atomic_def.inc>
#include <clc/integer/gentype.inc>
