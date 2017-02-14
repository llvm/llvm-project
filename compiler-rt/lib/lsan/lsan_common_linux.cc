//=-- lsan_common_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality. Linux-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#include "lsan_common.h"

#if CAN_SANITIZE_LEAKS && SANITIZER_LINUX
#include <link.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __lsan {

static const char kLinkerName[] = "ld";

static char linker_placeholder[sizeof(LoadedModule)] ALIGNED(64);
static LoadedModule *linker = nullptr;

static bool IsLinker(const char* full_name) {
  return LibraryNameIs(full_name, kLinkerName);
}

__attribute__((tls_model("initial-exec")))
THREADLOCAL int disable_counter;
bool DisabledInThisThread() { return disable_counter > 0; }
void DisableInThisThread() { disable_counter++; }
void EnableInThisThread() {
  if (disable_counter == 0) {
    DisableCounterUnderflow();
  }
  disable_counter--;
}

void InitializePlatformSpecificModules() {
  ListOfModules modules;
  modules.init();
  for (LoadedModule &module : modules) {
    if (!IsLinker(module.full_name())) continue;
    if (linker == nullptr) {
      linker = reinterpret_cast<LoadedModule *>(linker_placeholder);
      *linker = module;
      module = LoadedModule();
    } else {
      VReport(1, "LeakSanitizer: Multiple modules match \"%s\". "
              "TLS will not be handled correctly.\n", kLinkerName);
      linker->clear();
      linker = nullptr;
      return;
    }
  }
  VReport(1, "LeakSanitizer: Dynamic linker not found. "
             "TLS will not be handled correctly.\n");
}

static int ProcessGlobalRegionsCallback(struct dl_phdr_info *info, size_t size,
                                        void *data) {
  Frontier *frontier = reinterpret_cast<Frontier *>(data);
  for (uptr j = 0; j < info->dlpi_phnum; j++) {
    const ElfW(Phdr) *phdr = &(info->dlpi_phdr[j]);
    // We're looking for .data and .bss sections, which reside in writeable,
    // loadable segments.
    if (!(phdr->p_flags & PF_W) || (phdr->p_type != PT_LOAD) ||
        (phdr->p_memsz == 0))
      continue;
    uptr begin = info->dlpi_addr + phdr->p_vaddr;
    uptr end = begin + phdr->p_memsz;
    uptr allocator_begin = 0, allocator_end = 0;
    GetAllocatorGlobalRange(&allocator_begin, &allocator_end);
    if (begin <= allocator_begin && allocator_begin < end) {
      CHECK_LE(allocator_begin, allocator_end);
      CHECK_LE(allocator_end, end);
      if (begin < allocator_begin)
        ScanRangeForPointers(begin, allocator_begin, frontier, "GLOBAL",
                             kReachable);
      if (allocator_end < end)
        ScanRangeForPointers(allocator_end, end, frontier, "GLOBAL",
                             kReachable);
    } else {
      ScanRangeForPointers(begin, end, frontier, "GLOBAL", kReachable);
    }
  }
  return 0;
}

// Scans global variables for heap pointers.
void ProcessGlobalRegions(Frontier *frontier) {
  if (!flags()->use_globals) return;
  dl_iterate_phdr(ProcessGlobalRegionsCallback, frontier);
}

static uptr GetCallerPC(u32 stack_id, StackDepotReverseMap *map) {
  CHECK(stack_id);
  StackTrace stack = map->Get(stack_id);
  // The top frame is our malloc/calloc/etc. The next frame is the caller.
  if (stack.size >= 2)
    return stack.trace[1];
  return 0;
}

struct ProcessPlatformAllocParam {
  Frontier *frontier;
  StackDepotReverseMap *stack_depot_reverse_map;
  bool skip_linker_allocations;
};

// ForEachChunk callback. Identifies unreachable chunks which must be treated as
// reachable. Marks them as reachable and adds them to the frontier.
static void ProcessPlatformSpecificAllocationsCb(uptr chunk, void *arg) {
  CHECK(arg);
  ProcessPlatformAllocParam *param =
      reinterpret_cast<ProcessPlatformAllocParam *>(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() != kReachable && m.tag() != kIgnored) {
    u32 stack_id = m.stack_trace_id();
    uptr caller_pc = 0;
    if (stack_id > 0)
      caller_pc = GetCallerPC(stack_id, param->stack_depot_reverse_map);
    // If caller_pc is unknown, this chunk may be allocated in a coroutine. Mark
    // it as reachable, as we can't properly report its allocation stack anyway.
    if (caller_pc == 0 || (param->skip_linker_allocations &&
                           linker->containsAddress(caller_pc))) {
      m.set_tag(kReachable);
      param->frontier->push_back(chunk);
    }
  }
}

// Handles dynamically allocated TLS blocks by treating all chunks allocated
// from ld-linux.so as reachable.
// Dynamic TLS blocks contain the TLS variables of dynamically loaded modules.
// They are allocated with a __libc_memalign() call in allocate_and_init()
// (elf/dl-tls.c). Glibc won't tell us the address ranges occupied by those
// blocks, but we can make sure they come from our own allocator by intercepting
// __libc_memalign(). On top of that, there is no easy way to reach them. Their
// addresses are stored in a dynamically allocated array (the DTV) which is
// referenced from the static TLS. Unfortunately, we can't just rely on the DTV
// being reachable from the static TLS, and the dynamic TLS being reachable from
// the DTV. This is because the initial DTV is allocated before our interception
// mechanism kicks in, and thus we don't recognize it as allocated memory. We
// can't special-case it either, since we don't know its size.
// Our solution is to include in the root set all allocations made from
// ld-linux.so (which is where allocate_and_init() is implemented). This is
// guaranteed to include all dynamic TLS blocks (and possibly other allocations
// which we don't care about).
void ProcessPlatformSpecificAllocations(Frontier *frontier) {
  StackDepotReverseMap stack_depot_reverse_map;
  ProcessPlatformAllocParam arg;
  arg.frontier = frontier;
  arg.stack_depot_reverse_map = &stack_depot_reverse_map;
  arg.skip_linker_allocations =
      flags()->use_tls && flags()->use_ld_allocations && linker != nullptr;
  ForEachChunk(ProcessPlatformSpecificAllocationsCb, &arg);
}

struct DoStopTheWorldParam {
  StopTheWorldCallback callback;
  void *argument;
};

static int DoStopTheWorldCallback(struct dl_phdr_info *info, size_t size,
                                  void *data) {
  DoStopTheWorldParam *param = reinterpret_cast<DoStopTheWorldParam *>(data);
  StopTheWorld(param->callback, param->argument);
  return 1;
}

// LSan calls dl_iterate_phdr() from the tracer task. This may deadlock: if one
// of the threads is frozen while holding the libdl lock, the tracer will hang
// in dl_iterate_phdr() forever.
// Luckily, (a) the lock is reentrant and (b) libc can't distinguish between the
// tracer task and the thread that spawned it. Thus, if we run the tracer task
// while holding the libdl lock in the parent thread, we can safely reenter it
// in the tracer. The solution is to run stoptheworld from a dl_iterate_phdr()
// callback in the parent thread.
void DoStopTheWorld(StopTheWorldCallback callback, void *argument) {
  DoStopTheWorldParam param = {callback, argument};
  dl_iterate_phdr(DoStopTheWorldCallback, &param);
}

} // namespace __lsan

#endif // CAN_SANITIZE_LEAKS && SANITIZER_LINUX
