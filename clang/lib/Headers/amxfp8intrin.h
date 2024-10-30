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


/// Compute dot-product of brain-float8 (BF8) or hybrid-float8 (HF8)
///    floating-point pairs in tiles \a a and \a b, accumulating the
///    intermediate single-precision (32-bit) floating-point elements with
///    elements in \a dst, and store the 32-bit result back to tile \a dst.
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
#define _tile_dpbf8ps __builtin_ia32_tdpbf8ps

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
#define _tile_dpbhf8ps __builtin_ia32_tdpbhf8ps

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
#define _tile_dphbf8ps __builtin_ia32_tdphbf8ps

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
#define _tile_dphf8ps __builtin_ia32_tdphf8ps

#endif /* __x86_64__ */
#endif /* __AMXFP8INTRIN_H */
