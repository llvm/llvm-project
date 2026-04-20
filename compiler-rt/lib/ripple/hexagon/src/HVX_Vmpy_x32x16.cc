//==============================================================================
//
// Part of the Ripple vector library to support the HVX vmpyo x32x16
// instructions.
//
//==============================================================================
//
// HVX 32x16 odd-element multiply (vmpyo) runtime library.
//
// vmpyo multiplies the odd 16-bit half of each 32-bit lane of 'm' by the
// corresponding 16-bit lane of 'mc', with an optional left-shift by 1 and
// optional rounding/saturation.  The result fits in a 32-bit lane.
//
// Naming convention:
//   ripple_ew_pure_hvx_i32_vmpyo_i32i16_s1[_rnd]_sat
//     i32   - 32-bit signed output lane type
//     i32i16 - inputs: i32 vector (odd 16-bit half used) x i16 vector
//     s1    - left-shift result by 1 before storing
//     rnd   - add rounding bit before shift (optional)
//     sat   - saturate result to int32 range

#include "lib_func_attrib.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <ripple/zip.h>
#pragma once
#include <ripple_hvx.h>
#ifdef __cplusplus
extern "C" {
#endif

RIPPLE_INTRIN_INLINE
// out[i] = sat((m[i].hi16 * mc[i] << 1) + 0x8000) >> 16)
v32i32 ripple_ew_pure_hvx_i32_vmpyo_i32i16_s1_rnd_sat(v32i32 m, v32i32 mc) {
  return Q6_Vw_vmpyo_VwVh_s1_rnd_sat(m, mc);
}

// out[i] = sat(m[i].hi16 * mc[i] << 1)
RIPPLE_INTRIN_INLINE v32i32
ripple_ew_pure_hvx_i32_vmpyo_i32i16_s1_sat(v32i32 m, v32i32 mc) {
  return Q6_Vw_vmpyo_VwVh_s1_sat(m, mc);
}

#ifdef __cplusplus
}
#endif
