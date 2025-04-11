//===-- sanitizer_procmaps_aix.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (AIX-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_AIX
#  include <assert.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <sys/procfs.h>

#  include "sanitizer_common.h"
#  include "sanitizer_file.h"
#  include "sanitizer_procmaps.h"

namespace __sanitizer {

static int qsort_comp(const void *va, const void *vb) {
  const prmap_t *a = (const prmap_t *)va;
  const prmap_t *b = (const prmap_t *)vb;

  if (a->pr_vaddr < b->pr_vaddr)
    return -1;

  if (a->pr_vaddr > b->pr_vaddr)
    return 1;

  return 0;
}

static prmap_t *SortProcMapEntries(char *buffer) {
  prmap_t *begin = (prmap_t *)buffer;
  prmap_t *mapIter = begin;
  // The AIX procmap utility detects the end of the array of `prmap`s by finding
  // an entry where pr_size and pr_vaddr are both zero.
  while (mapIter->pr_size != 0 || mapIter->pr_vaddr != 0) ++mapIter;
  prmap_t *end = mapIter;

  size_t count = end - begin;
  size_t elemSize = sizeof(prmap_t);
  qsort(begin, count, elemSize, qsort_comp);

  return end;
}

void ReadProcMaps(ProcSelfMapsBuff *proc_maps) {
  uptr pid = internal_getpid();
  constexpr unsigned BUFFER_SIZE = 128;
  char filenameBuf[BUFFER_SIZE] = {};
  internal_snprintf(filenameBuf, BUFFER_SIZE, "/proc/%d/map", pid);
  if (!ReadFileToBuffer(filenameBuf, &proc_maps->data, &proc_maps->mmaped_size,
                        &proc_maps->len)) {
    proc_maps->data = nullptr;
    proc_maps->mmaped_size = 0;
    proc_maps->len = 0;
    proc_maps->mapEnd = nullptr;
    return;
  }

  proc_maps->mapEnd = SortProcMapEntries(proc_maps->data);
}

bool MemoryMappingLayout::Next(MemoryMappedSegment *segment) {
  if (Error())
    return false;  // simulate empty maps

  const prmap_t *mapIter = (const prmap_t *)data_.current;

  if (mapIter >= data_.proc_self_maps.mapEnd)
    return false;

  // Skip the kernel segment.
  if ((mapIter->pr_mflags & MA_TYPE_MASK) == MA_KERNTEXT)
    ++mapIter;

  if (mapIter >= data_.proc_self_maps.mapEnd)
    return false;

  segment->start = (uptr)mapIter->pr_vaddr;
  segment->end = segment->start + mapIter->pr_size;

  segment->protection = 0;
  uint32_t flags = mapIter->pr_mflags;
  if (flags & MA_READ)
    segment->protection |= kProtectionRead;
  if (flags & MA_WRITE)
    segment->protection |= kProtectionWrite;
  if (flags & MA_EXEC)
    segment->protection |= kProtectionExecute;

  // TODO FIXME why not PLIBTEXT?
  uint32_t type = mapIter->pr_mflags & MA_TYPE_MASK;
  if (type == MA_SLIBTEXT || type == MA_PLIBDATA)
    segment->protection |= kProtectionShared;

  if (segment->filename && mapIter->pr_pathoff) {
    uptr len;
    constexpr unsigned BUFFER_SIZE = 128;
    char objPath[BUFFER_SIZE] = {};
    // Use path /proc/<pid>/object/<object_id> to pass to the symbolizer.
    // TODO: Append the archive member name if it exists.
    internal_snprintf(objPath, BUFFER_SIZE, "/proc/%d/object/%s",
                      internal_getpid(), mapIter->pr_mapname);
    len = Min((uptr)internal_strlen(objPath), segment->filename_size - 1);
    internal_strncpy(segment->filename, objPath, len);
    segment->filename[len] = 0;

    // We don't have the full path to user libraries, so we use what we have
    // available as the display name.
    // TODO: Append the archive member name if it exists.
    const char *displayPath = data_.proc_self_maps.data + mapIter->pr_pathoff;
    len =
        Min((uptr)internal_strlen(displayPath), segment->displayname_size - 1);
    internal_strncpy(segment->displayname, displayPath, len);
    segment->displayname[len] = 0;
  } else if (segment->filename) {
    segment->filename[0] = 0;
    segment->displayname[0] = 0;
  }

  assert(mapIter->pr_off == 0 && "expect a zero offset into module.");
  segment->offset = 0;

  ++mapIter;
  data_.current = (const char *)mapIter;

  return true;
}

void MemoryMappingLayout::DumpListOfModules(
    InternalMmapVectorNoCtor<LoadedModule> *modules) {
  Reset();
  InternalMmapVector<char> module_name(kMaxPathLength);
  InternalMmapVector<char> module_displayname(kMaxPathLength);
  MemoryMappedSegment segment(module_name.data(), module_name.size(),
                              module_displayname.data(),
                              module_displayname.size());
  for (uptr i = 0; Next(&segment); i++) {
    const char *cur_name = segment.filename;
    if (cur_name[0] == '\0')
      continue;
    // Don't subtract 'cur_beg' from the first entry:
    // * If a binary is compiled w/o -pie, then the first entry in
    //   process maps is likely the binary itself (all dynamic libs
    //   are mapped higher in address space). For such a binary,
    //   instruction offset in binary coincides with the actual
    //   instruction address in virtual memory (as code section
    //   is mapped to a fixed memory range).
    // * If a binary is compiled with -pie, all the modules are
    //   mapped high at address space (in particular, higher than
    //   shadow memory of the tool), so the module can't be the
    //   first entry.
    uptr base_address = (i ? segment.start : 0) - segment.offset;
    LoadedModule cur_module;
    cur_module.set(cur_name, base_address);
    cur_module.setDisplayName(segment.displayname);
    segment.AddAddressRanges(&cur_module);
    modules->push_back(cur_module);
  }
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
