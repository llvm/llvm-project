//===-- sanitizer_mac.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// provides definitions for OSX-specific functions.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_MAC_H
#define SANITIZER_MAC_H

#include "sanitizer_common.h"
#include "sanitizer_platform.h"
#if SANITIZER_MAC
#include "sanitizer_posix.h"

namespace __sanitizer {

enum MacosVersion {
  MACOS_VERSION_UNINITIALIZED = 0,
  MACOS_VERSION_UNKNOWN,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION,
  MACOS_VERSION_MOUNTAIN_LION,
  MACOS_VERSION_MAVERICKS,
  MACOS_VERSION_YOSEMITE,
  MACOS_VERSION_UNKNOWN_NEWER
};

MacosVersion GetMacosVersion();

char **GetEnviron();

}  // namespace __sanitizer

extern "C" {
static char __crashreporter_info_buff__[kErrorMessageBufferSize] = {};
static const char *__crashreporter_info__ __attribute__((__used__)) =
  &__crashreporter_info_buff__[0];
asm(".desc ___crashreporter_info__, 0x10");
} // extern "C"
static BlockingMutex crashreporter_info_mutex(LINKER_INITIALIZED);

INLINE void CRAppendCrashLogMessage(const char *msg) {
  BlockingMutexLock l(&crashreporter_info_mutex);
  internal_strlcat(__crashreporter_info_buff__, msg,
                   sizeof(__crashreporter_info_buff__)); }

#endif  // SANITIZER_MAC
#endif  // SANITIZER_MAC_H
