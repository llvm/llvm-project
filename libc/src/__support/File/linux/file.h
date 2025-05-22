//===--- Linux specialization of the File data structure ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"

namespace LIBC_NAMESPACE {

FileIOResult linux_file_write(File *, const void *, size_t);
FileIOResult linux_file_read(File *, void *, size_t);
ErrorOr<long> linux_file_seek(File *, long, int);
int linux_file_close(File *);

class LinuxFile : public File {
  int fd;

public:
  constexpr LinuxFile(int file_descriptor, uint8_t *buffer, size_t buffer_size,
                      int buffer_mode, bool owned, File::ModeFlags modeflags)
      : File(&linux_file_write, &linux_file_read, &linux_file_seek,
             &linux_file_close, buffer, buffer_size, buffer_mode, owned,
             modeflags),
        fd(file_descriptor) {}

  int get_fd() const { return fd; }
};

} // namespace LIBC_NAMESPACE
