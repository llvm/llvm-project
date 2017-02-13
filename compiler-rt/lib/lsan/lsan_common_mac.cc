//=-- lsan_common_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality. Darwin-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#include "lsan_common.h"

#if CAN_SANITIZE_LEAKS && SANITIZER_MAC
namespace __lsan {

void InitializePlatformSpecificModules() {
  CHECK(0 && "unimplemented");
}

// Scans global variables for heap pointers.
void ProcessGlobalRegions(Frontier *frontier) {
  CHECK(0 && "unimplemented");
}

void ProcessPlatformSpecificAllocations(Frontier *frontier) {
  CHECK(0 && "unimplemented");
}

void DoStopTheWorld(StopTheWorldCallback callback, void *argument) {
  CHECK(0 && "unimplemented");
}

} // namespace __lsan

#endif // CAN_SANITIZE_LEAKS && SANITIZER_MAC
