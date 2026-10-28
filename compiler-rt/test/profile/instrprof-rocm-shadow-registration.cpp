// Check that callbacks following a failed table registration cannot modify
// the previous table's version or section handles. No GPU runtime is needed.
//
// Compile without instrumentation; the link supplies the profile library
// for the collector's writer and padding helpers.
// REQUIRES: linux
// RUN: %clangxx -I%S/../../lib -I%S/../../include -c %s -o %t.o
// RUN: %clangxx_profgen %t.o -pthread -ldl -o %t
// RUN: %run %t | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static int ReallocsUntilFailure = -1;
static void *controlledRealloc(void *Ptr, size_t Size) {
  if (ReallocsUntilFailure == 0) {
    ReallocsUntilFailure = -1;
    return nullptr;
  }
  if (ReallocsUntilFailure > 0)
    --ReallocsUntilFailure;
  return realloc(Ptr, Size);
}

// Include the production registration callbacks with allocation fault
// injection.
#define SANITIZER_COMMON_NO_REDEFINE_BUILTINS
#define realloc controlledRealloc
#include "profile/InstrProfilingPlatformROCm.cpp"
#undef realloc

// Disable runtime discovery: this test exercises host registration only.
namespace __interception {
bool DynamicLoaderAvailable() { return false; }
void *OpenLibrary(const char *) {
  assert(false);
  return nullptr;
}
void *LookupSymbol(void *, const char *) {
  assert(false);
  return nullptr;
}
bool InterceptFunction(const char *, uptr *, uptr, uptr) {
  assert(false);
  return false;
}
} // namespace __interception

namespace __prof_rocm {
void profRecordDrainedBounds(const void *, const void *, const void *) {}
int hsaRuntimeAvailable() { return 0; }
int drainDevicesViaHsa() { return 0; }
} // namespace __prof_rocm

struct Handles {
  int Table, Version, Sections[4];
};

static void registerDependents(Handles &H) {
  __llvm_profile_offload_register_version_shadow_variable(&H.Version);
  for (int &Section : H.Sections)
    __llvm_profile_offload_register_section_shadow_variable(&Section);
}

static void checkGroup(int Index, Handles &H) {
  const auto &Group = OffloadSectionShadowGroups[Index];
  assert(OffloadShadowVariables[Index] == &H.Table);
  assert(Group.VersionShadow == &H.Version);
  assert(Group.NumSections == 4 && Group.NumShadows == 1);
  assert(Group.Shadows[0].Data == &H.Sections[0]);
  assert(Group.Shadows[0].Counters == &H.Sections[1]);
  assert(Group.Shadows[0].UniformCounters == &H.Sections[2]);
  assert(Group.Shadows[0].Names == &H.Sections[3]);
}

int main() {
  Handles Tables[65] = {};
  // Dependent callbacks before the first table must also be ignored.
  registerDependents(Tables[64]);
  assert(NumShadowVariables == 0);

  for (int I = 0; I != 64; ++I) {
    __llvm_profile_offload_register_shadow_variable(&Tables[I].Table);
    registerDependents(Tables[I]);
  }
  assert(NumShadowVariables == 64);
  assert(CapShadowVariables == 64 && CapSectionShadowGroups == 64);

  for (int Allocation = 0; Allocation != 2; ++Allocation) {
    // First fail growing the table array, then fail growing the group array
    // after the table array has successfully grown.
    ReallocsUntilFailure = Allocation;
    __llvm_profile_offload_register_shadow_variable(&Tables[64].Table);
    assert(ReallocsUntilFailure == -1 && NumShadowVariables == 64);
    registerDependents(Tables[64]);
    for (int I = 0; I != 64; ++I)
      checkGroup(I, Tables[I]);
  }

  // A later successful table must start accepting its own callbacks again.
  __llvm_profile_offload_register_shadow_variable(&Tables[64].Table);
  registerDependents(Tables[64]);
  assert(NumShadowVariables == 65);
  for (int I = 0; I != 65; ++I)
    checkGroup(I, Tables[I]);
  puts("PASS");
}

// CHECK: PASS
