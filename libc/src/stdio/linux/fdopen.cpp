//===-- Implementation of fdopen --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fdopen.h"

#include "hdr/errno_macros.h"
#include "include/llvm-libc-macros/fcntl-macros.h"
#include "src/__support/File/linux/file.h"
#include "src/__support/OSUtil/fcntl.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(::FILE *, fdopen, (int fd, const char *mode)) {
  using ModeFlags = File::ModeFlags;
  ModeFlags modeflags = File::mode_flags(mode);
  if (modeflags == 0) {
    libc_errno = EINVAL;
    return nullptr;
  }

  int fd_flags = internal::fcntl(fd, F_GETFL);
  if (fd_flags == -1) {
    return nullptr;
  }

  using OpenMode = File::OpenMode;
  if (((fd_flags & O_ACCMODE) == O_RDONLY &&
       !(modeflags & static_cast<ModeFlags>(OpenMode::READ))) ||
      ((fd_flags & O_ACCMODE) == O_WRONLY &&
       !(modeflags & static_cast<ModeFlags>(OpenMode::WRITE)))) {
    libc_errno = EINVAL;
    return nullptr;
  }

  bool do_seek = false;
  if ((modeflags & static_cast<ModeFlags>(OpenMode::APPEND)) &&
      !(fd_flags & O_APPEND)) {
    do_seek = true;
    if (internal::fcntl(fd, F_SETFL,
                        reinterpret_cast<void *>(fd_flags | O_APPEND)) == -1) {
      return nullptr;
    }
  }

  uint8_t *buffer;
  {
    AllocChecker ac;
    buffer = new (ac) uint8_t[File::DEFAULT_BUFFER_SIZE];
    if (!ac) {
      libc_errno = ENOMEM;
      return nullptr;
    }
  }
  AllocChecker ac;
  auto *file = new (ac)
      LinuxFile(fd, buffer, File::DEFAULT_BUFFER_SIZE, _IOFBF, true, modeflags);
  if (!ac) {
    libc_errno = ENOMEM;
    return nullptr;
  }
  if (do_seek) {
    auto result = file->seek(0, SEEK_END);
    if (!result.has_value()) {
      libc_errno = result.error();
      free(file);
      return nullptr;
    }
  }

  return reinterpret_cast<::FILE *>(file);
}

} // namespace LIBC_NAMESPACE
