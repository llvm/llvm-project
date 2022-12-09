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
#include "trusty.h"

#include <stddef.h>
#include <string.h>

namespace scudo {

template <class Dest, class Source> inline Dest bit_cast(const Source &S) {
  static_assert(sizeof(Dest) == sizeof(Source), "");
  Dest D;
  memcpy(&D, &S, sizeof(D));
  return D;
}

inline constexpr uptr roundUpTo(uptr X, uptr Boundary) {
  return (X + Boundary - 1) & ~(Boundary - 1);
}

inline constexpr uptr roundDownTo(uptr X, uptr Boundary) {
  return X & ~(Boundary - 1);
}

inline constexpr bool isAligned(uptr X, uptr Alignment) {
  return (X & (Alignment - 1)) == 0;
}

template <class T> constexpr T Min(T A, T B) { return A < B ? A : B; }

template <class T> constexpr T Max(T A, T B) { return A > B ? A : B; }

template <class T> void Swap(T &A, T &B) {
  T Tmp = A;
  A = B;
  B = Tmp;
}

inline bool isPowerOfTwo(uptr X) { return (X & (X - 1)) == 0; }

inline uptr getMostSignificantSetBitIndex(uptr X) {
  DCHECK_NE(X, 0U);
  return SCUDO_WORDSIZE - 1U - static_cast<uptr>(__builtin_clzl(X));
}

inline uptr roundUpToPowerOfTwo(uptr Size) {
  DCHECK(Size);
  if (isPowerOfTwo(Size))
    return Size;
  const uptr Up = getMostSignificantSetBitIndex(Size);
  DCHECK_LT(Size, (1UL << (Up + 1)));
  DCHECK_GT(Size, (1UL << Up));
  return 1UL << (Up + 1);
}

inline uptr getLeastSignificantSetBitIndex(uptr X) {
  DCHECK_NE(X, 0U);
  return static_cast<uptr>(__builtin_ctzl(X));
}

inline uptr getLog2(uptr X) {
  DCHECK(isPowerOfTwo(X));
  return getLeastSignificantSetBitIndex(X);
}

inline u32 getRandomU32(u32 *State) {
  // ANSI C linear congruential PRNG (16-bit output).
  // return (*State = *State * 1103515245 + 12345) >> 16;
  // XorShift (32-bit output).
  *State ^= *State << 13;
  *State ^= *State >> 17;
  *State ^= *State << 5;
  return *State;
}

inline u32 getRandomModN(u32 *State, u32 N) {
  return getRandomU32(State) % N; // [0, N)
}

template <typename T> inline void shuffle(T *A, u32 N, u32 *RandState) {
  if (N <= 1)
    return;
  u32 State = *RandState;
  for (u32 I = N - 1; I > 0; I--)
    Swap(A[I], A[getRandomModN(&State, I + 1)]);
  *RandState = State;
}

// Hardware specific inlinable functions.

inline void yieldProcessor(u8 Count) {
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

extern uptr PageSizeCached;
uptr getPageSizeSlow();
inline uptr getPageSizeCached() {
  // Bionic uses a hardcoded value.
  if (SCUDO_ANDROID)
    return 4096U;
  if (LIKELY(PageSizeCached))
    return PageSizeCached;
  return getPageSizeSlow();
}

// Returns 0 if the number of CPUs could not be determined.
u32 getNumberOfCPUs();

const char *getEnv(const char *Name);

u64 GetRSS();

u64 getMonotonicTime();

u32 getThreadID();

// Our randomness gathering function is limited to 256 bytes to ensure we get
// as many bytes as requested, and avoid interruptions (on Linux).
constexpr uptr MaxRandomLength = 256U;
bool getRandom(void *Buffer, uptr Length, bool Blocking = false);

// Platform memory mapping functions.

#define MAP_ALLOWNOMEM (1U << 0)
#define MAP_NOACCESS (1U << 1)
#define MAP_RESIZABLE (1U << 2)
#define MAP_MEMTAG (1U << 3)

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

void setMemoryPermission(uptr Addr, uptr Size, uptr Flags,
                         MapPlatformData *Data = nullptr);

void releasePagesToOS(uptr BaseAddress, uptr Offset, uptr Size,
                      MapPlatformData *Data = nullptr);

// Internal map & unmap fatal error. This must not call map(). SizeIfOOM shall
// hold the requested size on an out-of-memory error, 0 otherwise.
void NORETURN dieOnMapUnmapError(uptr SizeIfOOM = 0);

// Logging related functions.

void setAbortMessage(const char *Message);

struct BlockInfo {
  uptr BlockBegin;
  uptr BlockSize;
  uptr RegionBegin;
  uptr RegionEnd;
};

enum class Option : u8 {
  ReleaseInterval,      // Release to OS interval in milliseconds.
  MemtagTuning,         // Whether to tune tagging for UAF or overflow.
  ThreadDisableMemInit, // Whether to disable automatic heap initialization and,
                        // where possible, memory tagging, on this thread.
  MaxCacheEntriesCount, // Maximum number of blocks that can be cached.
  MaxCacheEntrySize,    // Maximum size of a block that can be cached.
  MaxTSDsCount,         // Number of usable TSDs for the shared registry.
};

constexpr unsigned char PatternFillByte = 0xAB;

enum FillContentsMode {
  NoFill = 0,
  ZeroFill = 1,
  PatternOrZeroFill = 2 // Pattern fill unless the memory is known to be
                        // zero-initialized already.
};

} // namespace scudo

#endif // SCUDO_COMMON_H_
