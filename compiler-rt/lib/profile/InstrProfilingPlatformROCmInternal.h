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
// HSA-introspection drain (InstrProfilingPlatformROCmHSA.cpp). Not a public
// runtime header; everything lives in the __prof_rocm namespace with
// archive-internal linkage.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILE_INSTRPROFILINGPLATFORMROCMINTERNAL_H
#define PROFILE_INSTRPROFILINGPLATFORMROCMINTERNAL_H

#include <stddef.h>
#include <stdlib.h>

// Defined at global scope in InstrProfilingPlatformROCm.cpp. Forward-declared
// here (not redefined) so the HSA drain can name it in prototypes; the HSA path
// only ever passes a null group, so it never needs the full definition.
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
// of ElemSize bytes each. On success *Arr and *Cap are updated and the new tail
// slots [old cap, new cap) are zeroed; on allocation failure the existing array
// is left intact and -1 is returned (callers degrade gracefully rather than
// crash). *Arr is type-erased: pass the address of the typed array pointer,
// e.g. growArray((void **)&MI->TUs, &MI->CapTUs, ...).
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

// HIP/host-shadow helpers defined in InstrProfilingPlatformROCm.cpp and reused
// by the HSA drain.
int isVerboseMode();
void ensureHipLoaded();
// True once the loaded HIP runtime exposes hipMemcpy (device-to-host copies).
int hipMemcpyAvailable();
int memcpyDeviceToHost(void *Dst, const void *Src, size_t Size);
int processDeviceOffloadPrf(void *DeviceOffloadPrf, const char *Target,
                            const ::OffloadSectionShadowGroup *Sections);

#if defined(__linux__) && !defined(_WIN32)
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
