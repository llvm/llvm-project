/*===------------- amxfp8intrin.h - AMX intrinsics -*- C++ -*----------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <amxfp8intrin.h> directly; include <immintrin.h> instead."
#endif /* __IMMINTRIN_H */

#ifndef __AMXFP8INTRIN_H
#define __AMXFP8INTRIN_H
#ifdef __x86_64__

/// Peform the dot product of a BF8 value \a a by a BF8 value \a b accumulating
/// into a Single Precision (FP32) source/dest \a dst.
///
/// \headerfile <immintrin.h>
///
/// \code
/// void _tile_dpbf8ps (__tile dst, __tile a, __tile b)
/// \endcode
///
/// This intrinsic corresponds to the \c TDPBF8PS instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param a
///    The 1st source tile. Max size is 1024 Bytes.
/// \param b
///    The 2nd source tile. Max size is 1024 Bytes.
#define _tile_dpbf8ps(dst, a, b) __builtin_ia32_tdpbf8ps((dst), (a), (b))

/// Perform the dot product of a BF8 value \a a by an HF8 value \a b
/// accumulating into a Single Precision (FP32) source/dest \a dst.
///
/// \headerfile <immintrin.h>
///
/// \code
/// void _tile_dpbhf8ps (__tile dst, __tile a, __tile b)
/// \endcode
///
/// This intrinsic corresponds to the \c TDPBHF8PS instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param a
///    The 1st source tile. Max size is 1024 Bytes.
/// \param b
///    The 2nd source tile. Max size is 1024 Bytes.
#define _tile_dpbhf8ps(dst, a, b) __builtin_ia32_tdpbhf8ps((dst), (a), (b))

/// Perform the dot product of an HF8 value \a a by a BF8 value \a b
/// accumulating into a Single Precision (FP32) source/dest \a dst.
///
/// \headerfile <immintrin.h>
///
/// \code
/// void _tile_dphbf8ps (__tile dst, __tile a, __tile b)
/// \endcode
///
/// This intrinsic corresponds to the \c TDPHBF8PS instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param a
///    The 1st source tile. Max size is 1024 Bytes.
/// \param b
///    The 2nd source tile. Max size is 1024 Bytes.
#define _tile_dphbf8ps(dst, a, b) __builtin_ia32_tdphbf8ps((dst), (a), (b))

/// Perform the dot product of an HF8 value \a a by an HF8 value \a b
/// accumulating into a Single Precision (FP32) source/dest \a dst.
///
/// \headerfile <immintrin.h>
///
/// \code
/// void _tile_dphf8ps (__tile dst, __tile a, __tile b)
/// \endcode
///
/// This intrinsic corresponds to the \c TDPHF8PS instruction.
///
/// \param dst
///    The destination tile. Max size is 1024 Bytes.
/// \param a
///    The 1st source tile. Max size is 1024 Bytes.
/// \param b
///    The 2nd source tile. Max size is 1024 Bytes.
#define _tile_dphf8ps(dst, a, b) __builtin_ia32_tdphf8ps((dst), (a), (b))

#endif /* __x86_64__ */
#endif /* __AMXFP8INTRIN_H */
