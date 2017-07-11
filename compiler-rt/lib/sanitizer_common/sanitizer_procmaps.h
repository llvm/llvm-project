//===-- sanitizer_procmaps.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Information about the process mappings.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_PROCMAPS_H
#define SANITIZER_PROCMAPS_H

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

#if SANITIZER_FREEBSD || SANITIZER_LINUX
struct ProcSelfMapsBuff {
  char *data;
  uptr mmaped_size;
  uptr len;
};

// Reads process memory map in an OS-specific way.
void ReadProcMaps(ProcSelfMapsBuff *proc_maps);
#endif  // SANITIZER_FREEBSD || SANITIZER_LINUX

// Memory protection masks.
static const uptr kProtectionRead = 1;
static const uptr kProtectionWrite = 2;
static const uptr kProtectionExecute = 4;
static const uptr kProtectionShared = 8;

struct MemoryMappedSegment {
  MemoryMappedSegment(char *buff = nullptr, uptr size = 0)
      : filename(buff), filename_size(size) {}
  ~MemoryMappedSegment() {}

  bool IsReadable() { return protection & kProtectionRead; }
  bool IsWritable() { return protection & kProtectionWrite; }
  bool IsExecutable() { return protection & kProtectionExecute; }
  bool IsShared() { return protection & kProtectionShared; }

  uptr start;
  uptr end;
  uptr offset;
  char *filename;  // owned by caller
  uptr filename_size;
  uptr protection;
  ModuleArch arch;
  u8 uuid[kModuleUUIDSize];
};

class MemoryMappingLayout {
 public:
  explicit MemoryMappingLayout(bool cache_enabled);
  ~MemoryMappingLayout();
  bool Next(MemoryMappedSegment *segment);
  void Reset();
  // In some cases, e.g. when running under a sandbox on Linux, ASan is unable
  // to obtain the memory mappings. It should fall back to pre-cached data
  // instead of aborting.
  static void CacheMemoryMappings();

  // Adds all mapped objects into a vector.
  void DumpListOfModules(InternalMmapVector<LoadedModule> *modules);

 private:
  void LoadFromCache();

  // FIXME: Hide implementation details for different platforms in
  // platform-specific files.
# if SANITIZER_FREEBSD || SANITIZER_LINUX
  ProcSelfMapsBuff proc_self_maps_;
  const char *current_;

  // Static mappings cache.
  static ProcSelfMapsBuff cached_proc_self_maps_;
  static StaticSpinMutex cache_lock_;  // protects cached_proc_self_maps_.
# elif SANITIZER_MAC
  template <u32 kLCSegment, typename SegmentCommand>
  bool NextSegmentLoad(MemoryMappedSegment *segment);
  int current_image_;
  u32 current_magic_;
  u32 current_filetype_;
  ModuleArch current_arch_;
  u8 current_uuid_[kModuleUUIDSize];
  int current_load_cmd_count_;
  char *current_load_cmd_addr_;
  bool current_instrumented_;
# endif
};

typedef void (*fill_profile_f)(uptr start, uptr rss, bool file,
                               /*out*/uptr *stats, uptr stats_size);

// Parse the contents of /proc/self/smaps and generate a memory profile.
// |cb| is a tool-specific callback that fills the |stats| array containing
// |stats_size| elements.
void GetMemoryProfile(fill_profile_f cb, uptr *stats, uptr stats_size);

// Returns code range for the specified module.
bool GetCodeRangeForFile(const char *module, uptr *start, uptr *end);

bool IsDecimal(char c);
uptr ParseDecimal(const char **p);
bool IsHex(char c);
uptr ParseHex(const char **p);

}  // namespace __sanitizer

#endif  // SANITIZER_PROCMAPS_H
