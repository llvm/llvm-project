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

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "lsan_allocator.h"

#include <pthread.h>

#include <mach/mach.h>

namespace __lsan {

typedef struct {
  int disable_counter;
  u32 current_thread_id;
  AllocatorCache cache;
} thread_local_data_t;

static pthread_key_t key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

// The main thread destructor requires the current thread id,
// so we can't destroy it until it's been used and reset to invalid tid
void restore_tid_data(void *ptr) {
  thread_local_data_t *data = (thread_local_data_t *)ptr;
  if (data->current_thread_id != kInvalidTid)
    pthread_setspecific(key, data);
}

static void make_tls_key() {
  CHECK_EQ(pthread_key_create(&key, restore_tid_data), 0);
}

static thread_local_data_t *get_tls_val(bool alloc) {
  pthread_once(&key_once, make_tls_key);

  thread_local_data_t *ptr = (thread_local_data_t *)pthread_getspecific(key);
  if (ptr == NULL && alloc) {
    ptr = (thread_local_data_t *)InternalAlloc(sizeof(*ptr));
    ptr->disable_counter = 0;
    ptr->current_thread_id = kInvalidTid;
    ptr->cache = AllocatorCache();
    pthread_setspecific(key, ptr);
  }

  return ptr;
}

bool DisabledInThisThread() {
  thread_local_data_t *data = get_tls_val(false);
  return data ? data->disable_counter > 0 : false;
}

void DisableInThisThread() { ++get_tls_val(true)->disable_counter; }

void EnableInThisThread() {
  int *disable_counter = &get_tls_val(true)->disable_counter;
  if (*disable_counter == 0) {
    DisableCounterUnderflow();
  }
  --*disable_counter;
}

u32 GetCurrentThread() {
  thread_local_data_t *data = get_tls_val(false);
  CHECK(data);
  return data->current_thread_id;
}

void SetCurrentThread(u32 tid) { get_tls_val(true)->current_thread_id = tid; }

AllocatorCache *GetAllocatorCache() { return &get_tls_val(true)->cache; }

LoadedModule *GetLinker() { return nullptr; }

// Required on Linux for initialization of TLS behavior, but should not be
// required on Darwin.
void InitializePlatformSpecificModules() {
  if (flags()->use_tls) {
    Report("use_tls=1 is not supported on Darwin.\n");
    Die();
  }
}

// Scans global variables for heap pointers.
void ProcessGlobalRegions(Frontier *frontier) {
  MemoryMappingLayout memory_mapping(false);
  InternalMmapVector<LoadedModule> modules(/*initial_capacity*/ 128);
  memory_mapping.DumpListOfModules(&modules);
  for (uptr i = 0; i < modules.size(); ++i) {
    // Even when global scanning is disabled, we still need to scan
    // system libraries for stashed pointers
    if (!flags()->use_globals && modules[i].instrumented()) continue;

    for (const __sanitizer::LoadedModule::AddressRange &range :
         modules[i].ranges()) {
      if (range.executable || !range.readable) continue;

      ScanGlobalRange(range.beg, range.end, frontier);
    }
  }
}

void ProcessPlatformSpecificAllocations(Frontier *frontier) {
  mach_port_name_t port;
  if (task_for_pid(mach_task_self(), internal_getpid(), &port)
      != KERN_SUCCESS) {
    return;
  }

  unsigned depth = 1;
  vm_size_t size = 0;
  vm_address_t address = 0;
  kern_return_t err = KERN_SUCCESS;
  mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;

  InternalMmapVector<RootRegion> const *root_regions = GetRootRegions();

  while (err == KERN_SUCCESS) {
    struct vm_region_submap_info_64 info;
    err = vm_region_recurse_64(port, &address, &size, &depth,
                               (vm_region_info_t)&info, &count);

    uptr end_address = address + size;

    // libxpc stashes some pointers in the Kernel Alloc Once page,
    // make sure not to report those as leaks.
    if (info.user_tag == VM_MEMORY_OS_ALLOC_ONCE) {
      ScanRangeForPointers(address, end_address, frontier, "GLOBAL",
                           kReachable);
    }

    // This additional root region scan is required on Darwin in order to
    // detect root regions contained within mmap'd memory regions, because
    // the Darwin implementation of sanitizer_procmaps traverses images
    // as loaded by dyld, and not the complete set of all memory regions.
    //
    // TODO(fjricci) - remove this once sanitizer_procmaps_mac has the same
    // behavior as sanitizer_procmaps_linux and traverses all memory regions
    if (flags()->use_root_regions) {
      for (uptr i = 0; i < root_regions->size(); i++) {
        ScanRootRegion(frontier, (*root_regions)[i], address, end_address,
                       info.protection);
      }
    }

    address = end_address;
  }
}

void DoStopTheWorld(StopTheWorldCallback callback, void *argument) {
  StopTheWorld(callback, argument);
}

} // namespace __lsan

#endif // CAN_SANITIZE_LEAKS && SANITIZER_MAC
