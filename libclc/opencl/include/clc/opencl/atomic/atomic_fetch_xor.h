//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_XOR_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_XOR_H__

#define __CLC_FUNCTION atomic_fetch_xor

#define __CLC_BODY <clc/opencl/atomic/atomic_decl.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_FUNCTION

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_FETCH_XOR_H__
