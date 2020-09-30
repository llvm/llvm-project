/*===-------------- keylocker_wide_intrin.h - KL_WIDE Intrinsics ------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <keylocker_wide_intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef _KEYLOCKERINTRIN_WIDE_H
#define _KEYLOCKERINTRIN_WIDE_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__, __target__("kl,widekl"),\
                 __min_vector_width__(128)))

/// Encrypt __idata[0] to __idata[7] using 128-bit AES key indicated by handle
/// at __h and store each resultant block back from __odata to __odata+7. And
/// return the affected ZF flag status.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> AESENCWIDE128KL </c> instructions.
///
/// \operation
/// Handle := MEM[__h+383:__h]
/// IllegalHandle := ( HandleReservedBitSet (Handle[383:0]) ||
///                    (Handle[127:0] AND (CPL > 0)) ||
///                    Handle[255:128] ||
///                    HandleKeyType (Handle[383:0]) != HANDLE_KEY_TYPE_AES128 )
/// IF (IllegalHandle)
///   ZF := 1
/// ELSE
///   (UnwrappedKey, Authentic) := UnwrapKeyAndAuthenticate384 (Handle[383:0], IWKey)
///   IF Authentic == 0
///     ZF := 1
///   ELSE
///     FOR i := 0 to 7
///       __odata[i] := AES128Encrypt (__idata[i], UnwrappedKey)
///     ENDFOR
///     ZF := 0
///   FI
/// FI
/// dst := ZF
/// OF := 0
/// SF := 0
/// AF := 0
/// PF := 0
/// CF := 0
/// \endoperation
static __inline__ unsigned char __DEFAULT_FN_ATTRS
_mm_aesencwide128kl_u8(__m128i __odata[8], const __m128i __idata[8], const void* __h) {
  return __builtin_ia32_aesencwide128kl(__h,
                                        __odata,
                                        __odata + 1,
                                        __odata + 2,
                                        __odata + 3,
                                        __odata + 4,
                                        __odata + 5,
                                        __odata + 6,
                                        __odata + 7,
                                        __idata[0],
                                        __idata[1],
                                        __idata[2],
                                        __idata[3],
                                        __idata[4],
                                        __idata[5],
                                        __idata[6],
                                        __idata[7]);
}

/// Encrypt __idata[0] to __idata[7] using 256-bit AES key indicated by handle
/// at __h and store each resultant block back from __odata to __odata+7. And
/// return the affected ZF flag status.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> AESENCWIDE256KL </c> instructions.
///
/// \operation
/// Handle[511:0] := MEM[__h+511:__h]
/// IllegalHandle := ( HandleReservedBitSet (Handle[511:0]) ||
///                    (Handle[127:0] AND (CPL > 0)) ||
///                    Handle[255:128] ||
///                    HandleKeyType (Handle[511:0]) != HANDLE_KEY_TYPE_AES512 )
/// IF (IllegalHandle)
///   ZF := 1
/// ELSE
///   (UnwrappedKey, Authentic) := UnwrapKeyAndAuthenticate512 (Handle[511:0], IWKey)
///   IF Authentic == 0
///     ZF := 1
///   ELSE
///     FOR i := 0 to 7
///       __odata[i] := AES256Encrypt (__idata[i], UnwrappedKey)
///     ENDFOR
///     ZF := 0
///   FI
/// FI
/// dst := ZF
/// OF := 0
/// SF := 0
/// AF := 0
/// PF := 0
/// CF := 0
/// \endoperation
static __inline__ unsigned char __DEFAULT_FN_ATTRS
_mm_aesencwide256kl_u8(__m128i __odata[8], const __m128i __idata[8], const void* __h) {
  return __builtin_ia32_aesencwide256kl(__h,
                                        __odata,
                                        __odata + 1,
                                        __odata + 2,
                                        __odata + 3,
                                        __odata + 4,
                                        __odata + 5,
                                        __odata + 6,
                                        __odata + 7,
                                        __idata[0],
                                        __idata[1],
                                        __idata[2],
                                        __idata[3],
                                        __idata[4],
                                        __idata[5],
                                        __idata[6],
                                        __idata[7]);
}

/// Decrypt __idata[0] to __idata[7] using 128-bit AES key indicated by handle
/// at __h and store each resultant block back from __odata to __odata+7. And
/// return the affected ZF flag status.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> AESDECWIDE128KL </c> instructions.
///
/// \operation
/// Handle[383:0] := MEM[__h+383:__h]
/// IllegalHandle := ( HandleReservedBitSet (Handle[383:0]) ||
///                    (Handle[127:0] AND (CPL > 0)) ||
///                    Handle[255:128] ||
///                    HandleKeyType (Handle) != HANDLE_KEY_TYPE_AES128 )
/// IF (IllegalHandle)
///   ZF := 1
/// ELSE
///   (UnwrappedKey, Authentic) := UnwrapKeyAndAuthenticate384 (Handle[383:0], IWKey)
///   IF Authentic == 0
///     ZF := 1
///   ELSE
///     FOR i := 0 to 7
///       __odata[i] := AES128Decrypt (__idata[i], UnwrappedKey)
///     ENDFOR
///     ZF := 0
///   FI
/// FI
/// dst := ZF
/// OF := 0
/// SF := 0
/// AF := 0
/// PF := 0
/// CF := 0
/// \endoperation
static __inline__ unsigned char __DEFAULT_FN_ATTRS
_mm_aesdecwide128kl_u8(__m128i __odata[8], const __m128i __idata[8], const void* __h) {
  return __builtin_ia32_aesdecwide128kl(__h,
                                        __odata,
                                        __odata + 1,
                                        __odata + 2,
                                        __odata + 3,
                                        __odata + 4,
                                        __odata + 5,
                                        __odata + 6,
                                        __odata + 7,
                                        __idata[0],
                                        __idata[1],
                                        __idata[2],
                                        __idata[3],
                                        __idata[4],
                                        __idata[5],
                                        __idata[6],
                                        __idata[7]);
}

/// Decrypt __idata[0] to __idata[7] using 256-bit AES key indicated by handle
/// at __h and store each resultant block back from __odata to __odata+7. And
/// return the affected ZF flag status.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> AESDECWIDE256KL </c> instructions.
///
/// \operation
/// Handle[511:0] := MEM[__h+511:__h]
/// IllegalHandle = ( HandleReservedBitSet (Handle[511:0]) ||
///                   (Handle[127:0] AND (CPL > 0)) ||
///                   Handle[255:128] ||
///                   HandleKeyType (Handle) != HANDLE_KEY_TYPE_AES512 )
/// If (IllegalHandle)
///   ZF := 1
/// ELSE
///   (UnwrappedKey, Authentic) := UnwrapKeyAndAuthenticate512 (Handle[511:0], IWKey)
///   IF Authentic == 0
///     ZF := 1
///   ELSE
///     FOR i := 0 to 7
///       __odata[i] := AES256Decrypt (__idata[i], UnwrappedKey)
///     ENDFOR
///     ZF := 0
///   FI
/// FI
/// dst := ZF
/// OF := 0
/// SF := 0
/// AF := 0
/// PF := 0
/// CF := 0
/// \endoperation
static __inline__ unsigned char __DEFAULT_FN_ATTRS
_mm_aesdecwide256kl_u8(__m128i __odata[8], const __m128i __idata[8], const void* __h) {
  return __builtin_ia32_aesdecwide256kl(__h,
                                        __odata,
                                        __odata + 1,
                                        __odata + 2,
                                        __odata + 3,
                                        __odata + 4,
                                        __odata + 5,
                                        __odata + 6,
                                        __odata + 7,
                                        __idata[0],
                                        __idata[1],
                                        __idata[2],
                                        __idata[3],
                                        __idata[4],
                                        __idata[5],
                                        __idata[6],
                                        __idata[7]);
}


#undef __DEFAULT_FN_ATTRS

#endif /* _KEYLOCKERINTRIN_WIDE_H */
