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

#  include "sanitizer_common.h"
#  include "sanitizer_procmaps.h"
#  include "sanitizer_file.h"

namespace __sanitizer {

static bool IsOneOf(char c, char c1, char c2) { return c == c1 || c == c2; }

void ReadProcMaps(ProcSelfMapsBuff *proc_maps) {
  uptr pid = internal_getpid();

  // The mapping in /proc/id/map is not ordered by address, this will hit some
  // issue when checking stack base and size. Howevern AIX procmap can generate
  // sorted ranges.
  char Command[100] = {};

  internal_snprintf(Command, 100, "procmap -qX %d", pid);
  // Open pipe to file
  __sanitizer_FILE *pipe = internal_popen(Command, "r");

  if (!pipe) {
    proc_maps->data = nullptr;
    proc_maps->mmaped_size = 0;
    proc_maps->len = 0;
    return;
  }

  char buffer[512] = {};

  InternalScopedString Data;
  while (fgets(buffer, 512, reinterpret_cast<FILE *>(pipe)) != nullptr)
    Data.Append(buffer);

  size_t MmapedSize = Data.length() * 4 / 3;
  void *VmMap = MmapOrDie(MmapedSize, "ReadProcMaps()");
  internal_memcpy(VmMap, Data.data(), Data.length());

  proc_maps->data = (char *)VmMap;
  proc_maps->mmaped_size = MmapedSize;
  proc_maps->len = Data.length();

  internal_pclose(pipe);
}

bool MemoryMappingLayout::Next(MemoryMappedSegment *segment) {
  if (Error())
    return false;  // simulate empty maps
  char *last = data_.proc_self_maps.data + data_.proc_self_maps.len;
  if (data_.current >= last)
    return false;
  char *next_line =
      (char *)internal_memchr(data_.current, '\n', last - data_.current);

  // Skip the first header line and the second kernel line
  // pid : binary name
  if (data_.current == data_.proc_self_maps.data) {
    data_.current = next_line + 1;
    next_line =
        (char *)internal_memchr(next_line + 1, '\n', last - data_.current);

    data_.current = next_line + 1;
    next_line =
        (char *)internal_memchr(next_line + 1, '\n', last - data_.current);
  }

  if (next_line == 0)
    next_line = last;

  // Skip the last line:
  // Total   533562K
  if (!IsHex(*data_.current))
    return false;

  // Example: 10000000  10161fd9  1415K r-x   s  MAINTEXT  151ed82  a.out
  segment->start = ParseHex(&data_.current);
  while (data_.current < next_line && *data_.current == ' ') data_.current++;

  segment->end = ParseHex(&data_.current);
  while (data_.current < next_line && *data_.current == ' ') data_.current++;

  // Ignore the size, we can get accurate size from end and start
  while (IsDecimal(*data_.current)) data_.current++;
  CHECK_EQ(*data_.current++, 'K');

  while (data_.current < next_line && *data_.current == ' ') data_.current++;
  segment->protection = 0;

  if (*data_.current++ == 'r')
    segment->protection |= kProtectionRead;
  CHECK(IsOneOf(*data_.current, '-', 'w'));
  if (*data_.current++ == 'w')
    segment->protection |= kProtectionWrite;
  CHECK(IsOneOf(*data_.current, '-', 'x'));
  if (*data_.current++ == 'x')
    segment->protection |= kProtectionExecute;

  // Ignore the PSIZE(s/m/L/H)
  while (data_.current < next_line && *data_.current == ' ') data_.current++;
  data_.current += 4;

  // Get the region TYPE
  while (data_.current < next_line && *data_.current == ' ') data_.current++;
  char Type[16] = {};
  uptr len = 0;
  while (*data_.current != ' ') Type[len++] = *data_.current++;
  Type[len] = 0;

  if (!internal_strcmp(Type, "SLIBTEXT") || !internal_strcmp(Type, "PLIBDATA"))
    segment->protection |= kProtectionShared;

  // Ignore the VSID
  while (data_.current < next_line && *data_.current == ' ') data_.current++;
  ParseHex(&data_.current);

  while (data_.current < next_line && *data_.current == ' ') data_.current++;

  if (segment->filename && data_.current != next_line) {
    if (!internal_strcmp(Type, "MAINDATA") ||
        !internal_strcmp(Type, "MAINTEXT")) {
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
      char *NameEnd = (char *)internal_memchr(data_.current, '[',
                                              next_line - data_.current);
      if (!NameEnd)
        NameEnd = next_line - 1;

      uptr len = Min((uptr)(NameEnd - data_.current),
                     segment->filename_size - 1);
      internal_strncpy(segment->filename, data_.current, len);
      segment->filename[len] = 0;

      // AIX procmap does not print full name for user's library , however when
      // use llvm-symbolizer, it requires the library must be with full name.
      if ((!internal_strcmp(Type, "SLIBTEXT") ||
           !internal_strcmp(Type, "PLIBDATA")) &&
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

  segment->offset = 0;

  data_.current = next_line + 1;

  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
