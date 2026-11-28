//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux specialization of the File data structure.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_LINUX_FILE_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_LINUX_FILE_H

#include "hdr/func/free.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/File/file.h"
#include "src/__support/File/file_mode.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

class LinuxFile;

struct FilePlatform {
  using StreamBase = LinuxFile;

  static FileIOResult write(File *f, const void *data, size_t size);
  static FileIOResult read(File *f, void *buf, size_t size);
  static ErrorOr<off_t> seek(File *f, off_t offset, int whence);
  static int close(File *f);
};

class LinuxFile : public File {
  int fd;

  friend struct FilePlatform;

protected:
  constexpr LinuxFile(DispatchKind dispatch_kind, uint8_t *buffer,
                      size_t buffer_size, int buffer_mode, bool owned,
                      FileMode mode, int file_descriptor = -1)
      : File(dispatch_kind, buffer, buffer_size, buffer_mode, owned, mode),
        fd(file_descriptor) {}

  // Closes the underlying file descriptor. Returns 0 on success, or an error
  // code. This does not free the file object.
  int close_fd();

public:
  constexpr LinuxFile(int file_descriptor, uint8_t *buffer, size_t buffer_size,
                      int buffer_mode, bool owned, FileMode mode)
      : File(buffer, buffer_size, buffer_mode, owned, mode),
        fd(file_descriptor) {}

  // Allocates a FileT owning a new default-size buffer. FileT must be
  // LinuxFile or a subclass with LinuxFile's constructor signature.
  template <typename FileT = LinuxFile>
  LIBC_INLINE static ErrorOr<FileT *> create_buffered(int fd, FileMode mode) {
    static_assert(cpp::is_base_of_v<LinuxFile, FileT>,
                  "FileT must be LinuxFile or derive from it");
    uint8_t *buffer;
    {
      AllocChecker ac;
      buffer = new (ac) uint8_t[DEFAULT_BUFFER_SIZE];
      if (!ac)
        return Error(ENOMEM);
    }
    AllocChecker ac;
    auto *file = new (ac) FileT(fd, buffer, DEFAULT_BUFFER_SIZE, _IOFBF,
                                /*owned=*/true, mode);
    if (!ac) {
      // Use free instead of delete[] to avoid a GCC 12 ICE in the waccess pass.
      free(buffer);
      return Error(ENOMEM);
    }
    return file;
  }

  int get_fd() const { return fd; }
  void set_fd(int new_fd) { fd = new_fd; }

  int reopen_unlocked(const char *path, const char *mode);
};

using CustomFile = CustomFileImpl<LinuxFile>;
using CookieFile = CookieFileImpl<LinuxFile>;

// Create a File object and associate it with a fd.
ErrorOr<LinuxFile *> create_file_from_fd(int fd, const char *mode);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_LINUX_FILE_H
