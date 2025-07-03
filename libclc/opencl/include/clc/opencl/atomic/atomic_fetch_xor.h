//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define FUNCTION atomic_fetch_xor

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/integer/gentype.inc>

#undef FUNCTION
