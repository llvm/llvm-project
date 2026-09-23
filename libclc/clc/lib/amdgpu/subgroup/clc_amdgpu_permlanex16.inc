//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Typed wrappers around __builtin_amdgcn_permlanex16 as it does not natively
// handle non-integer types.
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR)

#if (defined(__CLC_GENSIZE) && __CLC_GENSIZE <= 32) ||                         \
    (defined(__CLC_FPSIZE) && __CLC_FPSIZE <= 32)

static _CLC_OVERLOAD __CLC_GENTYPE __clc_amdgpu_permlanex16(__CLC_GENTYPE old,
                                                            __CLC_GENTYPE src,
                                                            uint sel_lo,
                                                            uint sel_hi) {
  uint o = (uint)__CLC_AS_U_GENTYPE(old);
  uint s = (uint)__CLC_AS_U_GENTYPE(src);
  uint r = __builtin_amdgcn_permlanex16(o, s, sel_lo, sel_hi, 0, 0);
  return __CLC_AS_GENTYPE((__CLC_U_GENTYPE)r);
}

#elif (defined(__CLC_GENSIZE) && __CLC_GENSIZE == 64) ||                       \
    (defined(__CLC_FPSIZE) && __CLC_FPSIZE == 64)

static _CLC_OVERLOAD __CLC_GENTYPE __clc_amdgpu_permlanex16(__CLC_GENTYPE old,
                                                            __CLC_GENTYPE src,
                                                            uint sel_lo,
                                                            uint sel_hi) {
  uint2 o = __builtin_astype((__CLC_U_GENTYPE)old, uint2);
  uint2 s = __builtin_astype((__CLC_U_GENTYPE)src, uint2);
  uint2 r;
  r.lo = __builtin_amdgcn_permlanex16(o.lo, s.lo, sel_lo, sel_hi, 0, 0);
  r.hi = __builtin_amdgcn_permlanex16(o.hi, s.hi, sel_lo, sel_hi, 0, 0);
  return (__CLC_GENTYPE)__builtin_astype(r, __CLC_U_GENTYPE);
}

#endif

#endif // __CLC_SCALAR
