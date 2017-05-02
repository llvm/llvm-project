//===-- sanitizer_coverage_libcdep_new.cc ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Sanitizer Coverage Controller for Trace PC Guard.

#include "sancov_flags.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_symbolizer.h"

using namespace __sanitizer;

using AddressRange = LoadedModule::AddressRange;

namespace __sancov {
namespace {

static const u64 Magic64 = 0xC0BFFFFFFFFFFF64ULL;
static const u64 Magic32 = 0xC0BFFFFFFFFFFF32ULL;
static const u64 Magic = SANITIZER_WORDSIZE == 64 ? Magic64 : Magic32;

static fd_t OpenFile(const char* path) {
  error_t err;
  fd_t fd = OpenFile(path, WrOnly, &err);
  if (fd == kInvalidFd)
    Report("SanitizerCoverage: failed to open %s for writing (reason: %d)\n",
           path, err);
  return fd;
}

static void GetCoverageFilename(char* path, const char* name,
                                const char* extension) {
  CHECK(name);
  internal_snprintf(path, kMaxPathLength, "%s/%s.%zd.%s",
                    common_flags()->coverage_dir, name, internal_getpid(),
                    extension);
}

static void WriteModuleCoverage(char* file_path, const char* module_name,
                                const uptr* pcs, uptr len) {
  GetCoverageFilename(file_path, StripModuleName(module_name), "sancov");
  fd_t fd = OpenFile(file_path);
  WriteToFile(fd, &Magic, sizeof(Magic));
  WriteToFile(fd, pcs, len * sizeof(*pcs));
  CloseFile(fd);
  Printf("SanitizerCoverage: %s %zd PCs written\n", file_path, len);
}

static void SanitizerDumpCoverage(const uptr* unsorted_pcs, uptr len) {
  if (!len) return;

  char* file_path = static_cast<char*>(InternalAlloc(kMaxPathLength));
  char* module_name = static_cast<char*>(InternalAlloc(kMaxPathLength));
  uptr* pcs = static_cast<uptr*>(InternalAlloc(len * sizeof(uptr)));

  internal_memcpy(pcs, unsorted_pcs, len * sizeof(uptr));
  SortArray(pcs, len);

  bool module_found = false;
  uptr last_base = 0;
  uptr module_start_idx = 0;

  for (uptr i = 0; i < len; ++i) {
    const uptr pc = pcs[i];
    if (!pc) continue;

    if (!__sanitizer_get_module_and_offset_for_pc(pc, nullptr, 0, &pcs[i])) {
      Printf("ERROR: bad pc %x\n", pc);
      continue;
    }
    uptr module_base = pc - pcs[i];

    if (module_base != last_base || !module_found) {
      if (module_found) {
        WriteModuleCoverage(file_path, module_name, &pcs[module_start_idx],
                            i - module_start_idx);
      }

      last_base = module_base;
      module_start_idx = i;
      module_found = true;
      __sanitizer_get_module_and_offset_for_pc(pc, module_name, kMaxPathLength,
                                               &pcs[i]);
    }
  }

  if (module_found) {
    WriteModuleCoverage(file_path, module_name, &pcs[module_start_idx],
                        len - module_start_idx);
  }

  InternalFree(file_path);
  InternalFree(module_name);
  InternalFree(pcs);
}

// Collects trace-pc guard coverage.
// This class relies on zero-initialization.
class TracePcGuardController {
 public:
  void Initialize() {
    CHECK(!initialized);

    initialized = true;
    InitializeSancovFlags();

    pc_vector.Initialize(0);
  }

  void InitTracePcGuard(u32* start, u32* end) {
    if (!initialized) Initialize();
    CHECK(!*start);
    CHECK_NE(start, end);

    u32 i = pc_vector.size();
    for (u32* p = start; p < end; p++) *p = ++i;
    pc_vector.resize(i);
  }

  void TracePcGuard(u32* guard, uptr pc) {
    atomic_uint32_t* guard_ptr = reinterpret_cast<atomic_uint32_t*>(guard);
    u32 idx = atomic_exchange(guard_ptr, 0, memory_order_relaxed);
    if (!idx) return;
    // we start indices from 1.
    pc_vector[idx - 1] = pc;
  }

  void Dump() {
    if (!initialized || !common_flags()->coverage) return;
    __sanitizer_dump_coverage(pc_vector.data(), pc_vector.size());
  }

 private:
  bool initialized;
  InternalMmapVectorNoCtor<uptr> pc_vector;
};

static TracePcGuardController pc_guard_controller;

}  // namespace
}  // namespace __sancov

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_dump_coverage(  // NOLINT
    const uptr* pcs, uptr len) {
  return __sancov::SanitizerDumpCoverage(pcs, len);
}

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_trace_pc_guard, u32* guard) {
  if (!*guard) return;
  __sancov::pc_guard_controller.TracePcGuard(guard, GET_CALLER_PC() - 1);
}

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_cov_trace_pc_guard_init,
                             u32* start, u32* end) {
  if (start == end || *start) return;
  __sancov::pc_guard_controller.InitTracePcGuard(start, end);
}

SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_dump_trace_pc_guard_coverage() {
  __sancov::pc_guard_controller.Dump();
}
}  // extern "C"
