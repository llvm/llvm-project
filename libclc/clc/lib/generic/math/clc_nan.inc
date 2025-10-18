//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 1

#if __CLC_FPSIZE == 64
#define __CLC_IMPL_FUNCTION __builtin_nan
#elif __CLC_FPSIZE == 32
#define __CLC_IMPL_FUNCTION __builtin_nanf
#elif __CLC_FPSIZE == 16
#define __CLC_IMPL_FUNCTION __builtin_nanf16
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_nan(__CLC_U_GENTYPE code) {
  return __CLC_IMPL_FUNCTION("");
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_nan(__CLC_S_GENTYPE code) {
  return __CLC_IMPL_FUNCTION("");
}

#undef __CLC_IMPL_FUNCTION

#endif // __CLC_VECSIZE_OR_1 == 1
