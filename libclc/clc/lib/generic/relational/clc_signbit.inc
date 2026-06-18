//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 1
#define __CLC_RETTYPE __CLC_INTN
#else
#define __CLC_RETTYPE __CLC_S_GENTYPE
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_RETTYPE __clc_signbit(__CLC_GENTYPE x) {
#if __CLC_VECSIZE_OR_1 == 1
  return (__CLC_INTN)(__CLC_AS_U_GENTYPE(x) >> (__CLC_FPSIZE - 1));
#else
  return __CLC_AS_S_GENTYPE(x) >> (__CLC_FPSIZE - 1);
#endif
}

#undef __CLC_RETTYPE
