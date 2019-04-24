//===-- common.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_COMMON_H_
#define SCUDO_COMMON_H_

#include "internal_defs.h"

#include "fuchsia.h"
#include "linux.h"

#include <stddef.h>
#include <string.h>

namespace scudo {

template <class Dest, class Source> INLINE Dest bit_cast(const Source &S) {
  COMPILER_CHECK(sizeof(Dest) == sizeof(Source));
  Dest D;
  memcpy(&D, &S, sizeof(D));
  return D;
}

INLINE constexpr uptr roundUpTo(uptr X, uptr Boundary) {
  return (X + Boundary - 1) & ~(Boundary - 1);
}

INLINE constexpr uptr roundDownTo(uptr X, uptr Boundary) {
  return X & ~(Boundary - 1);
}

INLINE constexpr bool isAligned(uptr X, uptr Alignment) {
  return (X & (Alignment - 1)) == 0;
}

template <class T> constexpr T Min(T A, T B) { return A < B ? A : B; }

template <class T> constexpr T Max(T A, T B) { return A > B ? A : B; }

template <class T> void Swap(T &A, T &B) {
  T Tmp = A;
  A = B;
  B = Tmp;
}

INLINE bool isPowerOfTwo(uptr X) { return (X & (X - 1)) == 0; }

INLINE uptr getMostSignificantSetBitIndex(uptr X) {
  DCHECK_NE(X, 0U);
  return SCUDO_WORDSIZE - 1U - static_cast<uptr>(__builtin_clzl(X));
}

INLINE uptr roundUpToPowerOfTwo(uptr Size) {
  DCHECK(Size);
  if (isPowerOfTwo(Size))
    return Size;
  const uptr Up = getMostSignificantSetBitIndex(Size);
  DCHECK_LT(Size, (1UL << (Up + 1)));
  DCHECK_GT(Size, (1UL << Up));
  return 1UL << (Up + 1);
}

INLINE uptr getLeastSignificantSetBitIndex(uptr X) {
  DCHECK_NE(X, 0U);
  return static_cast<uptr>(__builtin_ctzll(X));
}

INLINE uptr getLog2(uptr X) {
  DCHECK(isPowerOfTwo(X));
  return getLeastSignificantSetBitIndex(X);
}

INLINE u32 getRandomU32(u32 *State) {
  // ANSI C linear congruential PRNG (16-bit output).
  // return (*State = *State * 1103515245 + 12345) >> 16;
  // XorShift (32-bit output).
  *State ^= *State << 13;
  *State ^= *State >> 17;
  *State ^= *State << 5;
  return *State;
}

INLINE u32 getRandomModN(u32 *State, u32 N) {
  return getRandomU32(State) % N; // [0, N)
}

template <typename T> INLINE void shuffle(T *A, u32 N, u32 *RandState) {
  if (N <= 1)
    return;
  u32 State = *RandState;
  for (u32 I = N - 1; I > 0; I--)
    Swap(A[I], A[getRandomModN(&State, I + 1)]);
  *RandState = State;
}

// Hardware specific inlinable functions.

INLINE void yieldProcessor(u8 Count) {
#if defined(__i386__) || defined(__x86_64__)
  __asm__ __volatile__("" ::: "memory");
  for (u8 I = 0; I < Count; I++)
    __asm__ __volatile__("pause");
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("" ::: "memory");
  for (u8 I = 0; I < Count; I++)
    __asm__ __volatile__("yield");
#endif
  __asm__ __volatile__("" ::: "memory");
}

// Platform specific functions.

void yieldPlatform();

extern uptr PageSizeCached;
uptr getPageSizeSlow();
INLINE uptr getPageSizeCached() {
  if (LIKELY(PageSizeCached))
    return PageSizeCached;
  return getPageSizeSlow();
}

u32 getNumberOfCPUs();

const char *getEnv(const char *Name);

u64 getMonotonicTime();

// Our randomness gathering function is limited to 256 bytes to ensure we get
// as many bytes as requested, and avoid interruptions (on Linux).
constexpr uptr MaxRandomLength = 256U;
bool getRandom(void *Buffer, uptr Length, bool Blocking = false);

// Platform memory mapping functions.

#define MAP_ALLOWNOMEM (1U << 0)
#define MAP_NOACCESS (1U << 1)
#define MAP_RESIZABLE (1U << 2)

// Our platform memory mapping use is restricted to 3 scenarios:
// - reserve memory at a random address (MAP_NOACCESS);
// - commit memory in a previously reserved space;
// - commit memory at a random address.
// As such, only a subset of parameters combinations is valid, which is checked
// by the function implementation. The Data parameter allows to pass opaque
// platform specific data to the function.
// Returns nullptr on error or dies if MAP_ALLOWNOMEM is not specified.
void *map(void *Addr, uptr Size, const char *Name, uptr Flags = 0,
          MapPlatformData *Data = nullptr);

// Indicates that we are getting rid of the whole mapping, which might have
// further consequences on Data, depending on the platform.
#define UNMAP_ALL (1U << 0)

void unmap(void *Addr, uptr Size, uptr Flags = 0,
           MapPlatformData *Data = nullptr);

void releasePagesToOS(uptr BaseAddress, uptr Offset, uptr Size,
                      MapPlatformData *Data = nullptr);

// Internal map & unmap fatal error. This must not call map().
void NORETURN dieOnMapUnmapError(bool OutOfMemory = false);

// Logging related functions.

void setAbortMessage(const char *Message);

} // namespace scudo

#endif // SCUDO_COMMON_H_
