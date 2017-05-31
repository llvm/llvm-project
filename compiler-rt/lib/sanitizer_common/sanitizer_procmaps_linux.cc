//===-- sanitizer_procmaps_linux.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (Linux-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

void ReadProcMaps(ProcSelfMapsBuff *proc_maps) {
  CHECK(ReadFileToBuffer("/proc/self/maps", &proc_maps->data,
                         &proc_maps->mmaped_size, &proc_maps->len));
}

static bool IsOneOf(char c, char c1, char c2) {
  return c == c1 || c == c2;
}

bool MemoryMappingLayout::Next(uptr *start, uptr *end, uptr *offset,
                               char filename[], uptr filename_size,
                               uptr *protection) {
  char *last = proc_self_maps_.data + proc_self_maps_.len;
  if (current_ >= last) return false;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  if (!protection) protection = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  // Example: 08048000-08056000 r-xp 00000000 03:0c 64593   /foo/bar
  *start = ParseHex(&current_);
  CHECK_EQ(*current_++, '-');
  *end = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  CHECK(IsOneOf(*current_, '-', 'r'));
  *protection = 0;
  if (*current_++ == 'r')
    *protection |= kProtectionRead;
  CHECK(IsOneOf(*current_, '-', 'w'));
  if (*current_++ == 'w')
    *protection |= kProtectionWrite;
  CHECK(IsOneOf(*current_, '-', 'x'));
  if (*current_++ == 'x')
    *protection |= kProtectionExecute;
  CHECK(IsOneOf(*current_, 's', 'p'));
  if (*current_++ == 's')
    *protection |= kProtectionShared;
  CHECK_EQ(*current_++, ' ');
  *offset = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ':');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  while (IsDecimal(*current_))
    current_++;
  // Qemu may lack the trailing space.
  // https://github.com/google/sanitizers/issues/160
  // CHECK_EQ(*current_++, ' ');
  // Skip spaces.
  while (current_ < next_line && *current_ == ' ')
    current_++;
  // Fill in the filename.
  uptr i = 0;
  while (current_ < next_line) {
    if (filename && i < filename_size - 1)
      filename[i++] = *current_;
    current_++;
  }
  if (filename && i < filename_size)
    filename[i] = 0;
  current_ = next_line + 1;
  return true;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
