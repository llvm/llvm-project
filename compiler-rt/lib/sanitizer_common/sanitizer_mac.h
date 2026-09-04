//===-- sanitizer_mac.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// provides definitions for OSX-specific functions.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_APPLE_H
#define SANITIZER_APPLE_H

#include "sanitizer_common.h"
#include "sanitizer_platform.h"
#if SANITIZER_APPLE
#include "sanitizer_posix.h"

// The earliest (macOS-aligned) version that requires debug memory
#  define DARWIN_DEBUG_MEMORY_VERSION_FLOOR (MacosVersion(27, 0))

// VM size threshold threshold above which we will use debug memory (448GB)
#  define DARWIN_DEBUG_MEMORY_VMOFFSET_FLOOR 0x7000000000UL

// The lowest address which sanitizers should expect to map
// on embedded devices with debug memory (VM_MEMORY_DEBUG)
#  define DARWIN_DEBUG_MEMORY_START 0x40000000000UL

namespace __sanitizer {

struct MemoryMappingLayoutData {
  int current_image;
  u32 current_magic;
  u32 current_filetype;
  ModuleArch current_arch;
  u8 current_uuid[kModuleUUIDSize];
  int current_load_cmd_count;
  const char *current_load_cmd_addr;
  bool current_instrumented;
};

template <typename VersionType>
struct VersionBase {
  u16 major;
  u16 minor;

  VersionBase(u16 major, u16 minor) : major(major), minor(minor) {}

  bool operator>=(const VersionType &other) const {
    return major > other.major ||
           (major == other.major && minor >= other.minor);
  }
  bool operator<(const VersionType &other) const { return !(*this >= other); }
};

template <typename VersionType>
bool operator==(const VersionBase<VersionType> &self,
                const VersionBase<VersionType> &other) {
  return self.major == other.major && self.minor == other.minor;
}

struct MacosVersion : VersionBase<MacosVersion> {
  MacosVersion(u16 major, u16 minor) : VersionBase(major, minor) {}
};

struct DarwinKernelVersion : VersionBase<DarwinKernelVersion> {
  DarwinKernelVersion(u16 major, u16 minor) : VersionBase(major, minor) {}
};

struct ReservedRange {
  uptr beg, end;
};

MacosVersion GetMacosAlignedVersion();
DarwinKernelVersion GetDarwinKernelVersion();
void GetAppRanges(InternalMmapVector<ReservedRange>& ranges);

extern bool debug_region_activated;
ALWAYS_INLINE static bool DebugMemoryActive() { return debug_region_activated; }
bool ActivateDebugMemory();
uptr FindAvailableMemoryRange(uptr size, uptr alignment, uptr left_padding,
                              uptr* largest_gap_found, uptr* max_occupied_addr,
                              bool use_debug_vm);

char **GetEnviron();

void RestrictMemoryToMaxAddress(uptr max_address);
bool MemoryRangeIsKernelReserved(uptr range_start, uptr range_end);

using ThreadEventCallback = void (*)(uptr thread);
using ThreadCreateEventCallback = void (*)(uptr thread, bool gcd_worker);
struct ThreadEventCallbacks {
  ThreadCreateEventCallback create;
  ThreadEventCallback start;
  ThreadEventCallback terminate;
  ThreadEventCallback destroy;
};

void InstallPthreadIntrospectionHook(const ThreadEventCallbacks &callbacks);

}  // namespace __sanitizer

#endif  // SANITIZER_APPLE
#endif  // SANITIZER_APPLE_H
