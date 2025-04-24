//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This does not exist for fp16
#if __CLC_FPSIZE > 16
#ifndef __CLC_SCALAR

#if __CLC_VECSIZE == 3
#  define __CLC_OFFSET 4
#else
#  define __CLC_OFFSET __CLC_VECSIZE
#endif

  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __private);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __local);
  FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __global);

#undef __CLC_OFFSET
#else
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __private);
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __local);
  FUNC(, 1, 1, __CLC_GENTYPE, __CLC_SCALAR_GENTYPE, __global);
#endif
#endif
