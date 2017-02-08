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

#include <atomic>
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
}

using namespace __sanitizer;
using namespace __xray;

// When set to 'true' this means the XRay runtime has been initialised. We use
// the weak symbols defined above (__start_xray_inst_map and
// __stop_xray_instr_map) to initialise the instrumentation map that XRay uses
// for runtime patching/unpatching of instrumentation points.
//
// FIXME: Support DSO instrumentation maps too. The current solution only works
// for statically linked executables.
std::atomic<bool> XRayInitialized{false};

// This should always be updated before XRayInitialized is updated.
std::atomic<__xray::XRaySledMap> XRayInstrMap{};

// __xray_init() will do the actual loading of the current process' memory map
// and then proceed to look for the .xray_instr_map section/segment.
void __xray_init() XRAY_NEVER_INSTRUMENT {
  initializeFlags();
  if (__start_xray_instr_map == nullptr) {
    Report("XRay instrumentation map missing. Not initializing XRay.\n");
    return;
  }

  // Now initialize the XRayInstrMap global struct with the address of the
  // entries, reinterpreted as an array of XRaySledEntry objects. We use the
  // virtual pointer we have from the section to provide us the correct
  // information.
  __xray::XRaySledMap SledMap{};
  SledMap.Sleds = __start_xray_instr_map;
  SledMap.Entries = __stop_xray_instr_map - __start_xray_instr_map;
  XRayInstrMap.store(SledMap, std::memory_order_release);
  XRayInitialized.store(true, std::memory_order_release);

  if (flags()->patch_premain)
    __xray_patch();
}

__attribute__((section(".preinit_array"),
               used)) void (*__local_xray_preinit)(void) = __xray_init;
