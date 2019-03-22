/* ===-------- ia32intrin.h ---------------------------------------------------===
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

#ifndef __X86INTRIN_H
#error "Never use <ia32intrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __IA32INTRIN_H
#define __IA32INTRIN_H

/** Counts the number of bits in the source operand having a value of 1.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> POPCNT </c> instruction or a
 *  a sequence of arithmetic and logic ops to calculate it.
 *
 *  \param __A
 *     An unsigned 32-bit integer operand.
 *  \returns A 32-bit integer containing the number of bits with value 1 in the
 *     source operand.
 */
static __inline__ int __attribute__((__always_inline__, __nodebug__))
__popcntd(unsigned int __A)
{
  return __builtin_popcount(__A);
}

#define _popcnt32(A) __popcntd((A))

#ifdef __x86_64__
/** Counts the number of bits in the source operand having a value of 1.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> POPCNT </c> instruction or a
 *  a sequence of arithmetic and logic ops to calculate it.
 *
 *  \param __A
 *     An unsigned 64-bit integer operand.
 *  \returns A 64-bit integer containing the number of bits with value 1 in the
 *     source operand.
 */
static __inline__ long long __attribute__((__always_inline__, __nodebug__))
__popcntq(unsigned long long __A)
{
  return __builtin_popcountll(__A);
}

#define _popcnt64(A) __popcntq((A))
#endif /* __x86_64__ */

#ifdef __x86_64__
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__))
__readeflags(void)
{
  return __builtin_ia32_readeflags_u64();
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
__writeeflags(unsigned long long __f)
{
  __builtin_ia32_writeeflags_u64(__f);
}

#else /* !__x86_64__ */
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__readeflags(void)
{
  return __builtin_ia32_readeflags_u32();
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
__writeeflags(unsigned int __f)
{
  __builtin_ia32_writeeflags_u32(__f);
}
#endif /* !__x86_64__ */

/** Adds the unsigned integer operand to the CRC-32C checksum of the
 *     unsigned char operand.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> CRC32B </c> instruction.
 *
 *  \param __C
 *     An unsigned integer operand to add to the CRC-32C checksum of operand
 *     \a  __D.
 *  \param __D
 *     An unsigned 8-bit integer operand used to compute the CRC-32C checksum.
 *  \returns The result of adding operand \a __C to the CRC-32C checksum of
 *     operand \a __D.
 */
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("sse4.2")))
__crc32b(unsigned int __C, unsigned char __D)
{
  return __builtin_ia32_crc32qi(__C, __D);
}

/** Adds the unsigned integer operand to the CRC-32C checksum of the
 *     unsigned short operand.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> CRC32W </c> instruction.
 *
 *  \param __C
 *     An unsigned integer operand to add to the CRC-32C checksum of operand
 *     \a  __D.
 *  \param __D
 *     An unsigned 16-bit integer operand used to compute the CRC-32C checksum.
 *  \returns The result of adding operand \a __C to the CRC-32C checksum of
 *     operand \a __D.
 */
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("sse4.2")))
__crc32w(unsigned int __C, unsigned short __D)
{
  return __builtin_ia32_crc32hi(__C, __D);
}

/** Adds the unsigned integer operand to the CRC-32C checksum of the
 *     second unsigned integer operand.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> CRC32D </c> instruction.
 *
 *  \param __C
 *     An unsigned integer operand to add to the CRC-32C checksum of operand
 *     \a  __D.
 *  \param __D
 *     An unsigned 32-bit integer operand used to compute the CRC-32C checksum.
 *  \returns The result of adding operand \a __C to the CRC-32C checksum of
 *     operand \a __D.
 */
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("sse4.2")))
__crc32d(unsigned int __C, unsigned int __D)
{
  return __builtin_ia32_crc32si(__C, __D);
}

#ifdef __x86_64__
/** Adds the unsigned integer operand to the CRC-32C checksum of the
 *     unsigned 64-bit integer operand.
 *
 *  \headerfile <x86intrin.h>
 *
 *  This intrinsic corresponds to the <c> CRC32Q </c> instruction.
 *
 *  \param __C
 *     An unsigned integer operand to add to the CRC-32C checksum of operand
 *     \a  __D.
 *  \param __D
 *     An unsigned 64-bit integer operand used to compute the CRC-32C checksum.
 *  \returns The result of adding operand \a __C to the CRC-32C checksum of
 *     operand \a __D.
 */
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__, __target__("sse4.2")))
__crc32q(unsigned long long __C, unsigned long long __D)
{
  return __builtin_ia32_crc32di(__C, __D);
}
#endif /* __x86_64__ */

static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__))
__rdpmc(int __A) {
  return __builtin_ia32_rdpmc(__A);
}

/* __rdtscp */
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__))
__rdtscp(unsigned int *__A) {
  return __builtin_ia32_rdtscp(__A);
}

#define _rdtsc() __rdtsc()

#define _rdpmc(A) __rdpmc(A)

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_wbinvd(void) {
  __builtin_ia32_wbinvd();
}

static __inline__ unsigned char __attribute__((__always_inline__, __nodebug__))
__rolb(unsigned char __X, int __C) {
  return __builtin_rotateleft8(__X, __C);
}

static __inline__ unsigned char __attribute__((__always_inline__, __nodebug__))
__rorb(unsigned char __X, int __C) {
  return __builtin_rotateright8(__X, __C);
}

static __inline__ unsigned short __attribute__((__always_inline__, __nodebug__))
__rolw(unsigned short __X, int __C) {
  return __builtin_rotateleft16(__X, __C);
}

static __inline__ unsigned short __attribute__((__always_inline__, __nodebug__))
__rorw(unsigned short __X, int __C) {
  return __builtin_rotateright16(__X, __C);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__rold(unsigned int __X, int __C) {
  return __builtin_rotateleft32(__X, __C);
}

static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__))
__rord(unsigned int __X, int __C) {
  return __builtin_rotateright32(__X, __C);
}

#ifdef __x86_64__
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__))
__rolq(unsigned long long __X, int __C) {
  return __builtin_rotateleft64(__X, __C);
}

static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__))
__rorq(unsigned long long __X, int __C) {
  return __builtin_rotateright64(__X, __C);
}
#endif /* __x86_64__ */

#ifndef _MSC_VER
/* These are already provided as builtins for MSVC. */
/* Select the correct function based on the size of long. */
#ifdef __LP64__
#define _lrotl(a,b) __rolq((a), (b))
#define _lrotr(a,b) __rorq((a), (b))
#else
#define _lrotl(a,b) __rold((a), (b))
#define _lrotr(a,b) __rord((a), (b))
#endif
#define _rotl(a,b) __rold((a), (b))
#define _rotr(a,b) __rord((a), (b))
#endif // _MSC_VER

/* These are not builtins so need to be provided in all modes. */
#define _rotwl(a,b) __rolw((a), (b))
#define _rotwr(a,b) __rorw((a), (b))

#endif /* __IA32INTRIN_H */
