//===-- xray_init.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// XRay initialisation logic.
//===----------------------------------------------------------------------===//

#include <fcntl.h>
#include <strings.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_common.h"
#include "xray/xray_interface.h"
#include "xray_allocator.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"

extern "C" {
void __xray_init();
extern const XRaySledEntry __start_xray_instr_map[] __attribute__((weak));
extern const XRaySledEntry __stop_xray_instr_map[] __attribute__((weak));
extern const XRayFunctionSledIndex __start_xray_fn_idx[] __attribute__((weak));
extern const XRayFunctionSledIndex __stop_xray_fn_idx[] __attribute__((weak));

#if SANITIZER_APPLE
// HACK: This is a temporary workaround to make XRay build on
// Darwin, but it will probably not work at runtime.
const XRaySledEntry __start_xray_instr_map[] = {};
extern const XRaySledEntry __stop_xray_instr_map[] = {};
extern const XRayFunctionSledIndex __start_xray_fn_idx[] = {};
extern const XRayFunctionSledIndex __stop_xray_fn_idx[] = {};
#endif
}

using namespace __xray;

// When set to 'true' this means the XRay runtime has been initialised. We use
// the weak symbols defined above (__start_xray_inst_map and
// __stop_xray_instr_map) to initialise the instrumentation map that XRay uses
// for runtime patching/unpatching of instrumentation points.
atomic_uint8_t XRayInitialized{0};

// This should always be updated before XRayInitialized is updated.
SpinMutex XRayInstrMapMutex;

//  Contains maps for the main executable as well as DSOs.
XRaySledMap *XRayInstrMaps;

// Number of binary objects registered.
atomic_uint32_t XRayNumObjects{0};

// Global flag to determine whether the flags have been initialized.
atomic_uint8_t XRayFlagsInitialized{0};

// A mutex to allow only one thread to initialize the XRay data structures.
SpinMutex XRayInitMutex;

// Registers XRay sleds and trampolines coming from the main executable or one
// of the linked DSOs.
// Returns the object ID if registration is successful, -1 otherwise.
int32_t
__xray_register_sleds(const XRaySledEntry *SledsBegin,
                      const XRaySledEntry *SledsEnd,
                      const XRayFunctionSledIndex *FnIndexBegin,
                      const XRayFunctionSledIndex *FnIndexEnd, bool FromDSO,
                      XRayTrampolines Trampolines) XRAY_NEVER_INSTRUMENT {
  if (!SledsBegin || !SledsEnd) {
    Report("Invalid XRay sleds.\n");
    return -1;
  }
  XRaySledMap SledMap;
  SledMap.FromDSO = FromDSO;
  SledMap.Loaded = true;
  SledMap.Trampolines = Trampolines;
  SledMap.Sleds = SledsBegin;
  SledMap.Entries = SledsEnd - SledsBegin;
  if (FnIndexBegin != nullptr) {
    SledMap.SledsIndex = FnIndexBegin;
    SledMap.Functions = FnIndexEnd - FnIndexBegin;
  } else {
    size_t CountFunctions = 0;
    uint64_t LastFnAddr = 0;

    for (std::size_t I = 0; I < SledMap.Entries; I++) {
      const auto &Sled = SledMap.Sleds[I];
      const auto Function = Sled.function();
      if (Function != LastFnAddr) {
        CountFunctions++;
        LastFnAddr = Function;
      }
    }
    SledMap.SledsIndex = nullptr;
    SledMap.Functions = CountFunctions;
  }
  if (SledMap.Functions >= XRayMaxFunctions) {
    Report("Too many functions! Maximum is %ld\n", XRayMaxFunctions);
    return -1;
  }

  if (Verbosity())
    Report("Registering %d new functions!\n", SledMap.Functions);

  {
    SpinMutexLock Guard(&XRayInstrMapMutex);
    auto Idx = atomic_fetch_add(&XRayNumObjects, 1, memory_order_acq_rel);
    if (Idx >= XRayMaxObjects) {
      Report("Too many objects registered! Maximum is %ld\n", XRayMaxObjects);
      return -1;
    }
    XRayInstrMaps[Idx] = std::move(SledMap);
    return Idx;
  }
}

// __xray_init() will do the actual loading of the current process' memory map
// and then proceed to look for the .xray_instr_map section/segment.
void __xray_init() XRAY_NEVER_INSTRUMENT {
  SpinMutexLock Guard(&XRayInitMutex);
  // Short-circuit if we've already initialized XRay before.
  if (atomic_load(&XRayInitialized, memory_order_acquire))
    return;

  // XRAY is not compatible with PaX MPROTECT
  CheckMPROTECT();

  if (!atomic_load(&XRayFlagsInitialized, memory_order_acquire)) {
    initializeFlags();
    atomic_store(&XRayFlagsInitialized, true, memory_order_release);
  }

  if (__start_xray_instr_map == nullptr) {
    if (Verbosity())
      Report("XRay instrumentation map missing. Not initializing XRay.\n");
    return;
  }

  atomic_store(&XRayNumObjects, 0, memory_order_release);

  // Pre-allocation takes up approx. 5kB for XRayMaxObjects=64.
  XRayInstrMaps = allocateBuffer<XRaySledMap>(XRayMaxObjects);

  int MainBinaryId =
      __xray_register_sleds(__start_xray_instr_map, __stop_xray_instr_map,
                            __start_xray_fn_idx, __stop_xray_fn_idx, false, {});

  // The executable should always get ID 0.
  if (MainBinaryId != 0) {
    Report("Registering XRay sleds failed.\n");
    return;
  }

  atomic_store(&XRayInitialized, true, memory_order_release);

#ifndef XRAY_NO_PREINIT
  if (flags()->patch_premain)
    __xray_patch();
#endif
}

// Registers XRay sleds and trampolines of an instrumented DSO.
// Returns the object ID if registration is successful, -1 otherwise.
//
// Default visibility is hidden, so we have to explicitly make it visible to
// DSO.
SANITIZER_INTERFACE_ATTRIBUTE int32_t __xray_register_dso(
    const XRaySledEntry *SledsBegin, const XRaySledEntry *SledsEnd,
    const XRayFunctionSledIndex *FnIndexBegin,
    const XRayFunctionSledIndex *FnIndexEnd,
    XRayTrampolines Trampolines) XRAY_NEVER_INSTRUMENT {
  // Make sure XRay has been initialized in the main executable.
  __xray_init();

  if (__xray_num_objects() == 0) {
    if (Verbosity())
      Report("No XRay instrumentation map in main executable. Not initializing "
             "XRay for DSO.\n");
    return -1;
  }

  // Register sleds in global map.
  int ObjId = __xray_register_sleds(SledsBegin, SledsEnd, FnIndexBegin,
                                    FnIndexEnd, true, Trampolines);

#ifndef XRAY_NO_PREINIT
  if (ObjId >= 0 && flags()->patch_premain)
    __xray_patch_object(ObjId);
#endif

  return ObjId;
}

// Deregisters a DSO from the main XRay runtime.
// Called from the DSO-local runtime when the library is unloaded (e.g. if
// dlclose is called).
// Returns true if the object ID is valid and the DSO was successfully
// deregistered.
SANITIZER_INTERFACE_ATTRIBUTE bool
__xray_deregister_dso(int32_t ObjId) XRAY_NEVER_INSTRUMENT {

  if (!atomic_load(&XRayInitialized, memory_order_acquire)) {
    if (Verbosity())
      Report("XRay has not been initialized. Cannot deregister DSO.\n");
    return false;
  }

  if (ObjId <= 0 || static_cast<uint32_t>(ObjId) >= __xray_num_objects()) {
    if (Verbosity())
      Report("Can't deregister object with ID %d: ID is invalid.\n", ObjId);
    return false;
  }

  {
    SpinMutexLock Guard(&XRayInstrMapMutex);
    auto &Entry = XRayInstrMaps[ObjId];
    if (!Entry.FromDSO) {
      if (Verbosity())
        Report("Can't deregister object with ID %d: object does not correspond "
               "to a shared library.\n",
               ObjId);
      return false;
    }
    if (!Entry.Loaded) {
      if (Verbosity())
        Report("Can't deregister object with ID %d: object is not loaded.\n",
               ObjId);
      return true;
    }
    // Mark DSO as unloaded. No need to unpatch.
    Entry.Loaded = false;
  }

  if (Verbosity())
    Report("Deregistered object with ID %d.\n", ObjId);

  return true;
}

// FIXME: Make check-xray tests work on FreeBSD without
// SANITIZER_CAN_USE_PREINIT_ARRAY.
// See sanitizer_internal_defs.h where the macro is defined.
// Calling unresolved PLT functions in .preinit_array can lead to deadlock on
// FreeBSD but here it seems benign.
#if !defined(XRAY_NO_PREINIT) &&                                               \
    (SANITIZER_CAN_USE_PREINIT_ARRAY || SANITIZER_FREEBSD)
// Only add the preinit array initialization if the sanitizers can.
__attribute__((section(".preinit_array"),
               used)) void (*__local_xray_preinit)(void) = __xray_init;
#else
// If we cannot use the .preinit_array section, we should instead use dynamic
// initialisation.
__attribute__ ((constructor (0)))
static void __local_xray_dyninit() {
  __xray_init();
}
#endif
