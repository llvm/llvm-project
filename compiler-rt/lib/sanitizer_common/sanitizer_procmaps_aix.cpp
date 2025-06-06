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

  uint32_t type = mapIter->pr_mflags & MA_TYPE_MASK;
  if (type == MA_SLIBTEXT || type == MA_PLIBDATA)
    segment->protection |= kProtectionShared;

  if (segment->filename && mapIter->pr_pathoff) {
    uptr len;
    constexpr unsigned BUFFER_SIZE = 128;
    char objPath[BUFFER_SIZE] = {};
    // Use path /proc/<pid>/object/<object_id>
    // TODO: Pass a separate path from mapIter->pr_pathoff to display to the
    // user.
    // TODO: Append the archive member name if it exists.
    internal_snprintf(objPath, BUFFER_SIZE, "/proc/%d/object/%s",
                      internal_getpid(), mapIter->pr_mapname);
    len = Min((uptr)internal_strlen(objPath), segment->filename_size - 1);
    internal_strncpy(segment->filename, objPath, len);
    segment->filename[len] = 0;

  } else if (segment->filename) {
    segment->filename[0] = 0;
  }

  assert(mapIter->pr_off == 0 && "expect a zero offset into module.");
  segment->offset = 0;

  ++mapIter;
  data_.current = (const char *)mapIter;

  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
