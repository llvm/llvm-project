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
#  include <stddef.h>
#  include <stdint.h>

#  include "sanitizer_posix.h"

// macOS sanitizer report format for Crash Reporting

#  define LLVM_SANITIZER_V1_CATEGORY_MAXLEN 32
#  define LLVM_SANITIZER_V1_TYPE_MAXLEN 32
#  define LLVM_SANITIZER_V1_STACK_DESCRIPTION_MAXLEN 128
#  define LLVM_SANITIZER_V1_MAXSTACKS 4

#  define LLVM_SANITIZER_V1_STACK_TYPE_OTHER 0
#  define LLVM_SANITIZER_V1_STACK_TYPE_ALLOCATION 1
#  define LLVM_SANITIZER_V1_STACK_TYPE_DEALLOCATION 2

typedef struct {
  uint64_t thread_id;
  uint64_t time;
  uint32_t num_frames;
  uintptr_t frames[64];
} sanitizers_stack_trace_t;

// Structs for the global format shared between the LLVM compiler-rt
// runtimes, and ReportCrash.
typedef struct {
  uint32_t type;
  char description[LLVM_SANITIZER_V1_STACK_DESCRIPTION_MAXLEN];
  sanitizers_stack_trace_t stack;
} llvm_sanitizer_report_payload_stack_v1;

typedef struct {
  char category[LLVM_SANITIZER_V1_CATEGORY_MAXLEN];
  char type[LLVM_SANITIZER_V1_TYPE_MAXLEN];

  // These three fields may be NULL for non-heap errors
  uintptr_t fault_address;
  uintptr_t allocation_address;
  size_t allocation_size;

  // Number of backtraces (up to LLVM_SANITIZER_V1_MAXSTACKS) that follow
  uint16_t nstacks;
  llvm_sanitizer_report_payload_stack_v1 stacks[LLVM_SANITIZER_V1_MAXSTACKS];
} llvm_sanitizer_report_payload_v1;

typedef struct __attribute__((packed)) {
  uint16_t vers;
  union {
    llvm_sanitizer_report_payload_v1 v1;
  };
} llvm_sanitizer_report_payload;

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
void GetAppReservedRanges(InternalMmapVector<ReservedRange>& ranges);

char **GetEnviron();

void RestrictMemoryToMaxAddress(uptr max_address);

using ThreadEventCallback = void (*)(uptr thread);
using ThreadCreateEventCallback = void (*)(uptr thread, bool gcd_worker);
struct ThreadEventCallbacks {
  ThreadCreateEventCallback create;
  ThreadEventCallback start;
  ThreadEventCallback terminate;
  ThreadEventCallback destroy;
};

void InstallPthreadIntrospectionHook(const ThreadEventCallbacks &callbacks);

void GetDarwinStack(
    InternalMmapVector<llvm_sanitizer_report_payload_stack_v1>& stacks, int tid,
    const StackTrace* s, uint16_t type, const char* description,
    bool includeThreadName = true);

void SetCrashReporterGlobalForReport(
    const char* error_name, uptr fault_addr, uptr allocation_addr,
    uptr allocation_size,
    const InternalMmapVector<llvm_sanitizer_report_payload_stack_v1>& stacks);

}  // namespace __sanitizer

#endif  // SANITIZER_APPLE
#endif  // SANITIZER_APPLE_H
