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
#  include <stdio.h>
#  include <stdlib.h>
#  include <sys/procfs.h>

#  include "sanitizer_common.h"
#  include "sanitizer_file.h"
#  include "sanitizer_procmaps.h"

using namespace __sanitizer;

static int qsort_comp(const void* va, const void* vb) {
  auto* a = static_cast<const prmap_t*>(va);
  auto* b = static_cast<const prmap_t*>(vb);

  if (a->pr_vaddr < b->pr_vaddr)
    return -1;

  if (a->pr_vaddr > b->pr_vaddr)
    return 1;

  CHECK_EQ(a, b);
  return 0;
}

static prmap_t* SortProcMapEntries(char* buffer, uptr len) {
  prmap_t* begin = reinterpret_cast<prmap_t*>(buffer);
  const char* bufferEnd = buffer + len;
  prmap_t* mapIter = begin;
  // The AIX procmap utility detects the end of the array of `prmap`s by finding
  // an entry where pr_size and pr_vaddr are both zero.
  while (reinterpret_cast<char*>(mapIter) < bufferEnd &&
         (mapIter->pr_size != 0 || mapIter->pr_vaddr != 0))
    ++mapIter;
  prmap_t* end = mapIter;

  size_t count = end - begin;
  size_t elemSize = sizeof(prmap_t);
  qsort(begin, count, elemSize, qsort_comp);

  return end;
}

void __sanitizer::ReadProcMaps(ProcSelfMapsBuff* proc_maps) {
  uptr pid = internal_getpid();
  constexpr unsigned BUFFER_SIZE = 128;
  char filenameBuf[BUFFER_SIZE] = {};
  int filenameLen =
      internal_snprintf(filenameBuf, BUFFER_SIZE, "/proc/%d/map", pid);
  CHECK_GE(filenameLen, 0);
  CHECK_LT(filenameLen, static_cast<int>(BUFFER_SIZE));
  if (!ReadFileToBuffer(filenameBuf, &proc_maps->data, &proc_maps->mmaped_size,
                        &proc_maps->len)) {
    proc_maps->data = nullptr;
    proc_maps->mmaped_size = 0;
    proc_maps->len = 0;
    proc_maps->mapEnd = nullptr;
    return;
  }

  proc_maps->mapEnd = SortProcMapEntries(proc_maps->data, proc_maps->len);
}

bool __sanitizer::MemoryMappingLayout::Next(MemoryMappedSegment* segment) {
  if (Error())
    return false;  // simulate empty maps

  auto* mapIter = reinterpret_cast<const prmap_t*>(data_.current);

  if (mapIter >= data_.proc_self_maps.mapEnd)
    return false;

  // Skip the kernel segment.
  if ((mapIter->pr_mflags & MA_TYPE_MASK) == MA_KERNTEXT)
    ++mapIter;

  if (mapIter >= data_.proc_self_maps.mapEnd)
    return false;

  // The following has to be a C-style cast because the source type requires
  // static_cast in 32-bit mode and reinterpret_cast in 64-bit mode.
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

  // Handle filenames for loaded objects.
  // pr_pathoff is non-zero iff the map entry is for a loaded object.
  if (segment->filename && mapIter->pr_pathoff) {
    // Use path /proc/<pid>/object/<object_id>.
    // TODO: Pass a separate path from mapIter->pr_pathoff to display to the
    // user.
    // FIXME: Append the archive member name if it exists.
    int objPathLen = internal_snprintf(
        segment->filename, segment->filename_size, "/proc/%d/object/%s",
        internal_getpid(), mapIter->pr_mapname);
    if (objPathLen < 0)
      segment->filename[0] = 0;
  } else if (segment->filename) {
    segment->filename[0] = 0;
  }

  // AIX does not report the offset into any loaded modules. The offset into the
  // module is best handled by recording the segment type and having the
  // symbolizer determine the offset using that.
  // FIXME: Record the segment type.
  segment->offset = 0;

  ++mapIter;
  data_.current = reinterpret_cast<const char*>(mapIter);

  return true;
}

#endif  // SANITIZER_AIX
