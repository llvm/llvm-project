//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===
// Header to support Narrow shifting using HVX
//===------------------------------------------------------------------------===
//
// These functions perform narrowing shift operations on pairs of values.
// Each function takes two input values (`odd` and `even`) and a shift amount,
// then returns a packed result.
//
// Naming convention:
// - `sat`     = saturating shift
// - `rnd_sat` = rounding + saturating shift
// - `noshuff` = elementwise implementations, does not interleave the results.
//
// High-level behavior:
// narrow_shift(odd, even, shift) = pack(
//     cast_down(shift_right(even, shift)),
//     cast_down(shift_right(odd, shift))
// )
// where:
// - cast_down: converts 32-bit to 16-bit (or 16-bit to 8-bit), with saturation
// or truncation
// - shift_right: may truncate, or round
// - pack: combines two narrowed values into a single result:
//     - standard version: interleaves two 16-bit (or 8-bit) values into a
//     32-bit (or 16-bit) result, with `even` in the lower half and `odd` in the
//     upper half
//     - `noshuff` version: preserves the input order: `odd` result goes into
//     the upper half and `even` result goes into the lower half
//     Note: The `noshuff` version currently supports only HVX-native vector
//     sizes (i.e., 128B). It should be used exclusively within
//     ripple_parallel_full.
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// int32 → int16, packed into uint32 (int16 + int16)
int32_t hvx_narsh_i32toi16(int32_t odd, int32_t even, uint32_t shift);
int32_t hvx_narsh_sat_i32toi16(int32_t odd, int32_t even, uint32_t shift);
int32_t hvx_narsh_rnd_sat_i32toi16(int32_t odd, int32_t even, uint32_t shift);
int32_t hvx_narsh_i32toi16_noshuff(int32_t odd, int32_t even, uint32_t shift);
int32_t hvx_narsh_sat_i32toi16_noshuff(int32_t odd, int32_t even,
                                       uint32_t shift);
int32_t hvx_narsh_rnd_sat_i32toi16_noshuff(int32_t odd, int32_t even,
                                           uint32_t shift);

// int32 → uint16, packed into uint32 (uint16 + uint16)
uint32_t hvx_narsh_sat_i32tou16(int32_t odd, int32_t even, uint32_t shift);
uint32_t hvx_narsh_rnd_sat_i32tou16(int32_t odd, int32_t even, uint32_t shift);
uint32_t hvx_narsh_sat_i32tou16_noshuff(int32_t odd, int32_t even,
                                        uint32_t shift);
uint32_t hvx_narsh_rnd_sat_i32tou16_noshuff(int32_t odd, int32_t even,
                                            uint32_t shift);

// uint32 → uint16, packed into uint32 (uint16 + uint16)
uint32_t hvx_narsh_sat_u32tou16(uint32_t odd, uint32_t even, uint32_t shift);
uint32_t hvx_narsh_rnd_sat_u32tou16(uint32_t odd, uint32_t even,
                                    uint32_t shift);
uint32_t hvx_narsh_sat_u32tou16_noshuff(uint32_t odd, uint32_t even,
                                        uint32_t shift);
uint32_t hvx_narsh_rnd_sat_u32tou16_noshuff(uint32_t odd, uint32_t even,
                                            uint32_t shift);

// int16 → int8, packed into int16 (int8 + int8)
int16_t hvx_narsh_sat_i16toi8(int16_t odd, int16_t even, uint32_t shift);
int16_t hvx_narsh_rnd_sat_i16toi8(int16_t odd, int16_t even, uint32_t shift);
int16_t hvx_narsh_sat_i16toi8_noshuff(int16_t odd, int16_t even,
                                      uint32_t shift);
int16_t hvx_narsh_rnd_sat_i16toi8_noshuff(int16_t odd, int16_t even,
                                          uint32_t shift);

// int16 → uint8, packed into uint16 (uint8 + uint8)
uint16_t hvx_narsh_sat_i16tou8(int16_t odd, int16_t even, uint32_t shift);
uint16_t hvx_narsh_rnd_sat_i16tou8(int16_t odd, int16_t even, uint32_t shift);
uint16_t hvx_narsh_sat_i16tou8_noshuff(int16_t odd, int16_t even,
                                       uint32_t shift);
uint16_t hvx_narsh_rnd_sat_i16tou8_noshuff(int16_t odd, int16_t even,
                                           uint32_t shift);

// uint16 → uint8, packed into uint16 (uint8 + uint8)
uint16_t hvx_narsh_sat_u16tou8(uint16_t odd, uint16_t even, uint32_t shift);
uint16_t hvx_narsh_rnd_sat_u16tou8(uint16_t odd, uint16_t even, uint32_t shift);
uint16_t hvx_narsh_sat_u16tou8_noshuff(uint16_t odd, uint16_t even,
                                       uint32_t shift);
uint16_t hvx_narsh_rnd_sat_u16tou8_noshuff(uint16_t odd, uint16_t even,
                                           uint32_t shift);

#ifdef __cplusplus
}
#endif
