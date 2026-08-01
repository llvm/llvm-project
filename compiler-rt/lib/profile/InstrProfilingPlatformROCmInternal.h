//===- InstrProfilingPlatformROCmInternal.h - ROCm shared interface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private interface shared between the ROCm host-shadow drain
// (InstrProfilingPlatformROCm.cpp) and the Linux-only supplemental
// HSA-introspection drain (InstrProfilingPlatformROCmHSA.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef PROFILE_INSTRPROFILINGPLATFORMROCMINTERNAL_H
#define PROFILE_INSTRPROFILINGPLATFORMROCMINTERNAL_H

#include <stddef.h>
#include <stdlib.h>

// For prototype declarations
struct OffloadSectionShadowGroup;

namespace __prof_rocm {

// free()-based scope guard. Use .release() to transfer ownership.
struct UniqueFree {
  void *Ptr;
  explicit UniqueFree(void *P = nullptr) : Ptr(P) {}
  ~UniqueFree() { free(Ptr); }
  UniqueFree(const UniqueFree &) = delete;
  UniqueFree &operator=(const UniqueFree &) = delete;
  char *get() const { return static_cast<char *>(Ptr); }
  void reset(void *P) {
    free(Ptr);
    Ptr = P;
  }
  void *release() {
    void *P = Ptr;
    Ptr = nullptr;
    return P;
  }
};

// Grow a heap array (doubling from InitCap) to hold at least MinCount elements
// of ElemSize bytes each.
// Success: zero new memory, update pointer, return 0.
// Failure: return -1, data is still intact.
inline int growArray(void **Arr, int *Cap, int MinCount, int InitCap,
                     size_t ElemSize) {
  if (*Cap >= MinCount)
    return 0;
  int NewCap = *Cap ? *Cap : InitCap;
  while (NewCap < MinCount)
    NewCap *= 2;
  void *New = realloc(*Arr, (size_t)NewCap * ElemSize);
  if (!New)
    return -1;
  __builtin_memset((char *)New + (size_t)*Cap * ElemSize, 0,
                   (size_t)(NewCap - *Cap) * ElemSize);
  *Arr = New;
  *Cap = NewCap;
  return 0;
}

// Set of (data, counters, names) device section-bounds tuples that have already
// been drained. Both ROCm drains record here so each unique device counter set
// is written exactly once.
// See test/profile/instrprof-rocm-bounds-dedup.cpp.
struct ProfBoundsSet {
  struct Tuple {
    const void *Data;
    const void *Counters;
    const void *Names;
  };
  enum { kInitCap = 64 };

  Tuple *Items = nullptr;
  int Count = 0;
  int Cap = 0;

  // True iff this exact (Data, Counters, Names) tuple was already recorded. All
  // three fields must match: two code objects can share, e.g., a names section.
  bool contains(const void *D, const void *C, const void *N) const {
    for (int I = 0; I < Count; ++I)
      if (Items[I].Data == D && Items[I].Counters == C && Items[I].Names == N)
        return true;
    return false;
  }

  // Record a tuple unless already present. Returns true only when a new tuple
  // was added (false for a duplicate or when the growth failed under OOM).
  bool record(const void *D, const void *C, const void *N) {
    if (contains(D, C, N))
      return false;
    if (growArray((void **)&Items, &Cap, Count + 1, kInitCap, sizeof(*Items)))
      return false;
    Items[Count].Data = D;
    Items[Count].Counters = C;
    Items[Count].Names = N;
    ++Count;
    return true;
  }
};

// HIP/host-shadow helpers defined in InstrProfilingPlatformROCm.cpp and reused
// by the HSA drain.
int isVerboseMode();
void ensureHipLoaded();
// True once the loaded HIP runtime exposes hipMemcpy (device-to-host copies).
int hipMemcpyAvailable();
int memcpyDeviceToHost(void *Dst, const void *Src, size_t Size);
int processDeviceOffloadPrf(void *DeviceOffloadPrf, const char *Target,
                            const ::OffloadSectionShadowGroup *Sections);

#if defined(__linux__)
// Implemented in InstrProfilingPlatformROCmHSA.cpp.

// Record a drained section-bounds tuple so the supplemental HSA pass skips any
// code object the host-shadow path already drained.
void profRecordDrainedBounds(const void *Data, const void *Counters,
                             const void *Names);

// Walk every GPU agent's loaded executables via HSA and drain each
// __llvm_profile_sections table the host-shadow pass did not already handle.
int drainDevicesViaHsa(void);
#endif

} // namespace __prof_rocm

#endif // PROFILE_INSTRPROFILINGPLATFORMROCMINTERNAL_H
