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
#  include <stdlib.h>
#  include <stdio.h>
#  include <sys/procfs.h>

#  include "sanitizer_common.h"
#  include "sanitizer_procmaps.h"
#  include "sanitizer_file.h"

namespace __sanitizer {

static int qsort_comp(const void *va, const void * vb) {
  const prmap_t *a = (const prmap_t *)va;
  const prmap_t *b = (const prmap_t *)vb;

  if (a->pr_vaddr < b->pr_vaddr)
    return -1;

  if (a->pr_vaddr > b->pr_vaddr)
    return 1;

  return 0;
}

static prmap_t *SortProcMapEntries(char *buffer) {
  prmap_t *begin = (prmap_t*)buffer;
  prmap_t *mapIter = begin;
  // The AIX procmap utility detects the end of the array of `prmap`s by finding
  // an entry where pr_size and pr_vaddr are both zero.
  while (mapIter->pr_size != 0 || mapIter->pr_vaddr != 0)
    ++mapIter;
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
  if (!ReadFileToBuffer(filenameBuf, &proc_maps->data, &proc_maps->mmaped_size, &proc_maps->len)) {
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
    if (type == MA_MAINDATA || type == MA_MAINEXEC) {
      // AIX procmap does not print full name for the binary, however when using
      // llvm-symbolizer, it requires the binary must be with full name.
      const char *BinaryName = GetBinaryName();
      uptr len =
          Min((uptr)(internal_strlen(BinaryName)), segment->filename_size - 1);
      internal_strncpy(segment->filename, BinaryName, len);
      segment->filename[len] = 0;
    } else {
      // AIX library may exist as xxx.a[yyy.o], to find the path to xxx.a,
      // the [yyy.o] part needs to be removed.

      // TODO FIXME
      const char *pathPtr = data_.proc_self_maps.data + mapIter->pr_pathoff;
      uptr len = Min((uptr)internal_strlen(pathPtr),
                     segment->filename_size - 1);
      internal_strncpy(segment->filename, pathPtr, len);
      segment->filename[len] = 0;
      // AIX procmap does not print full name for user's library , however when
      // use llvm-symbolizer, it requires the library must be with full name.
      if ((type == MA_SLIBTEXT || type == MA_PLIBDATA) &&
          segment->filename[0] != '/') {
        // First check if the library is in the directory where the binary is
        // executed. On AIX, there is no need to put library in same dir with
        // the binary to path search envs.
        char *path = nullptr;
        char buf[kMaxPathLength];
        unsigned buf_len = kMaxPathLength;
        bool found = false;
        if ((path = internal_getcwd(buf, buf_len)) != nullptr) {
          // if the path is too long, don't do other search either.
          if (internal_strlen(path) > segment->filename_size - 1)
            found = true;
          else {
            internal_snprintf(
                buf + internal_strlen(path),
                segment->filename_size - 1 - internal_strlen(path), "/%s",
                segment->filename);
            if (FileExists(buf)) {
              uptr len =
                  Min((uptr)(internal_strlen(buf)), segment->filename_size - 1);
              internal_strncpy(segment->filename, buf, len);
              segment->filename[len] = 0;
              found = true;
            }
          }
        }
        if (!found) {
          const char *LibName =
              FindPathToBinaryOrLibrary(segment->filename, "LIBPATH");
          CHECK(LibName);
          uptr len =
              Min((uptr)(internal_strlen(LibName)), segment->filename_size - 1);
          internal_strncpy(segment->filename, LibName, len);
          segment->filename[len] = 0;
	  found = true;
        }
        CHECK(found);
      }
    }
  } else if (segment->filename) {
    segment->filename[0] = 0;
  }

  assert(mapIter->pr_off == 0 && "expect a zero offset into module.");
  segment->offset = 0;

  ++mapIter;
  data_.current = (const char*)mapIter;

  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
