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
#include "xray_flags.h"
#include "xray_interface_internal.h"

extern "C" {
extern void __xray_init();
extern const XRaySledEntry __start_xray_instr_map[] __attribute__((weak));
extern const XRaySledEntry __stop_xray_instr_map[] __attribute__((weak));
}

using namespace __xray;

// We initialize some global variables that pertain to specific sections of XRay
// data structures in the binary. We do this for the current process using
// /proc/curproc/map and make sure that we're able to get it. We signal failure
// via a global atomic boolean to indicate whether we've initialized properly.
//
std::atomic<bool> XRayInitialized{false};

// This should always be updated before XRayInitialized is updated.
std::atomic<__xray::XRaySledMap> XRayInstrMap{};

// __xray_init() will do the actual loading of the current process' memory map
// and then proceed to look for the .xray_instr_map section/segment.
void __xray_init() {
  InitializeFlags();
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
