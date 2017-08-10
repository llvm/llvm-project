//===-- xray_init.cc --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"

extern "C" {
void __xray_init();
extern const XRaySledEntry __start_xray_instr_map[] __attribute__((weak));
extern const XRaySledEntry __stop_xray_instr_map[] __attribute__((weak));
extern const XRayFunctionSledIndex __start_xray_fn_idx[] __attribute__((weak));
extern const XRayFunctionSledIndex __stop_xray_fn_idx[] __attribute__((weak));
}

using namespace __xray;

// When set to 'true' this means the XRay runtime has been initialised. We use
// the weak symbols defined above (__start_xray_inst_map and
// __stop_xray_instr_map) to initialise the instrumentation map that XRay uses
// for runtime patching/unpatching of instrumentation points.
//
// FIXME: Support DSO instrumentation maps too. The current solution only works
// for statically linked executables.
__sanitizer::atomic_uint8_t XRayInitialized{0};

// This should always be updated before XRayInitialized is updated.
__sanitizer::SpinMutex XRayInstrMapMutex;
XRaySledMap XRayInstrMap;

// Global flag to determine whether the flags have been initialized.
__sanitizer::atomic_uint8_t XRayFlagsInitialized{0};

// A mutex to allow only one thread to initialize the XRay data structures.
__sanitizer::SpinMutex XRayInitMutex;

// __xray_init() will do the actual loading of the current process' memory map
// and then proceed to look for the .xray_instr_map section/segment.
void __xray_init() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayInitMutex);
  // Short-circuit if we've already initialized XRay before.
  if (__sanitizer::atomic_load(&XRayInitialized,
                               __sanitizer::memory_order_acquire))
    return;

  if (!__sanitizer::atomic_load(&XRayFlagsInitialized,
                                __sanitizer::memory_order_acquire)) {
    initializeFlags();
    __sanitizer::atomic_store(&XRayFlagsInitialized, true,
                              __sanitizer::memory_order_release);
  }

  if (__start_xray_instr_map == nullptr) {
    if (Verbosity())
      Report("XRay instrumentation map missing. Not initializing XRay.\n");
    return;
  }

  {
    __sanitizer::SpinMutexLock Guard(&XRayInstrMapMutex);
    XRayInstrMap.Sleds = __start_xray_instr_map;
    XRayInstrMap.Entries = __stop_xray_instr_map - __start_xray_instr_map;
    XRayInstrMap.SledsIndex = __start_xray_fn_idx;
    XRayInstrMap.Functions = __stop_xray_fn_idx - __start_xray_fn_idx;
  }
  __sanitizer::atomic_store(&XRayInitialized, true,
                            __sanitizer::memory_order_release);

#ifndef XRAY_NO_PREINIT
  if (flags()->patch_premain)
    __xray_patch();
#endif
}

#ifndef XRAY_NO_PREINIT
__attribute__((section(".preinit_array"),
               used)) void (*__local_xray_preinit)(void) = __xray_init;
#endif
