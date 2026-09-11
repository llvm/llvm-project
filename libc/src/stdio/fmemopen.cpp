//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of fmemopen, a POSIX function.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fmemopen.h"

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/new.h"
#include "src/__support/File/file.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

class MemoryFile : public File {
  // The stdio buffer is separate from the memory used as the file's contents.
  // In particular, ungetc and setvbuf must not modify or free that memory.
  uint8_t stream_buffer[DEFAULT_BUFFER_SIZE];
  uint8_t *storage;
  size_t capacity;
  size_t position = 0;
  size_t end = 0;
  bool owns_storage;
  bool append;

  static FileIOResult memory_read(File *f, void *data, size_t size) {
    auto *mf = reinterpret_cast<MemoryFile *>(f);
    if (size == 0 || mf->position >= mf->end)
      return 0;
    size_t available = mf->end - mf->position;
    size_t count = cpp::min(size, available);
    inline_memcpy(data, mf->storage + mf->position, count);
    mf->position += count;
    return count;
  }

  static FileIOResult memory_write(File *f, const void *data, size_t size) {
    auto *mf = reinterpret_cast<MemoryFile *>(f);
    if (size == 0)
      return 0;
    size_t start = mf->append ? mf->end : mf->position;
    size_t available = mf->capacity - start;
    size_t count = cpp::min(size, available);
    // POSIX fflush and fseek specify ENOSPC for a full fmemopen buffer.
    if (count == 0)
      return {0, ENOSPC};
    inline_memcpy(mf->storage + start, data, count);
    mf->position = start + count;
    if (mf->position > mf->end) {
      mf->end = mf->position;
      // The terminator is not part of the file contents. All capacity bytes
      // may hold data, in which case there is no room for a terminator.
      if (mf->end < mf->capacity)
        mf->storage[mf->end] = '\0';
    }
    return {count, count < size ? ENOSPC : 0};
  }

  static ErrorOr<off_t> memory_seek(File *f, off_t offset, int whence) {
    auto *mf = reinterpret_cast<MemoryFile *>(f);
    size_t base;
    switch (whence) {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = mf->position;
      break;
    case SEEK_END:
      base = mf->end;
      break;
    default:
      return Error(EINVAL);
    }

    // The bounds fit in off_t since no object is larger than PTRDIFF_MAX, so
    // comparing in off_t needs no negation of offset or narrowing to size_t.
    const off_t min_offset = -static_cast<off_t>(base);
    const off_t max_offset = static_cast<off_t>(mf->capacity - base);
    if (offset < min_offset || offset > max_offset)
      return Error(EINVAL);
    size_t next = base + static_cast<size_t>(offset);
    mf->position = next;
    return static_cast<off_t>(next);
  }

  static int memory_close(File *f) {
    auto *mf = reinterpret_cast<MemoryFile *>(f);
    File::remove_file(mf);
    if (mf->owns_storage)
      delete[] mf->storage;
    delete mf;
    return 0;
  }

public:
  MemoryFile(uint8_t *storage, size_t capacity, bool owns_storage,
             ModeFlags mode)
      : File(&memory_write, &memory_read, &memory_seek, &memory_close,
             stream_buffer, sizeof(stream_buffer), _IOFBF, false, mode),
        storage(storage), capacity(capacity), owns_storage(owns_storage),
        append(mode & static_cast<ModeFlags>(OpenMode::APPEND)) {
    if (mode & static_cast<ModeFlags>(OpenMode::READ)) {
      end = capacity;
    } else if (mode & static_cast<ModeFlags>(OpenMode::WRITE)) {
      if (capacity != 0)
        storage[0] = '\0';
    } else if (!owns_storage) {
      while (end < capacity && storage[end] != '\0')
        ++end;
      position = end;
    }
  }
};

} // namespace

LLVM_LIBC_FUNCTION(::FILE *, fmemopen,
                   (void *__restrict buf, size_t max_size,
                    const char *__restrict mode)) {
  LIBC_CRASH_ON_NULLPTR(mode);
  // Use the same mode parser as fopen. Binary mode has no special effect.
  auto flags = File::mode_flags(mode);
  if (flags == 0) {
    libc_errno = EINVAL;
    return nullptr;
  }

  auto *storage = static_cast<uint8_t *>(buf);
  bool owns_storage = buf == nullptr;
  if (owns_storage && max_size != 0) {
    AllocChecker ac;
    storage = new (ac) uint8_t[max_size];
    if (!ac) {
      libc_errno = ENOMEM;
      return nullptr;
    }
  }

  AllocChecker ac;
  auto *file = new (ac) MemoryFile(storage, max_size, owns_storage, flags);
  if (!ac) {
    if (owns_storage)
      delete[] storage;
    libc_errno = ENOMEM;
    return nullptr;
  }
  File::add_file(file);
  return reinterpret_cast<::FILE *>(file);
}

} // namespace LIBC_NAMESPACE_DECL
