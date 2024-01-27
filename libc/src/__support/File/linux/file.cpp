//===--- Implementation of the Linux specialization of File ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"

#include "src/__support/File/file.h"

#include "src/__support/CPP/new.h"
#include "src/__support/File/linux/lseekImpl.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/errno/libc_errno.h"         // For error macros

#include <fcntl.h> // For mode_t and other flags to the open syscall
#include <stdio.h>
#include <sys/stat.h>    // For S_IS*, S_IF*, and S_IR* flags.
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE {

FileIOResult linux_file_write(File *f, const void *data, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_write, lf->get_fd(), data, size);
  if (ret < 0) {
    return {0, -ret};
  }
  return ret;
}

FileIOResult linux_file_read(File *f, void *buf, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_read, lf->get_fd(), buf, size);
  if (ret < 0) {
    return {0, -ret};
  }
  return ret;
}

ErrorOr<long> linux_file_seek(File *f, long offset, int whence) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  auto result = internal::lseekimpl(lf->get_fd(), offset, whence);
  if (!result.has_value())
    return result.error();
  return result.value();
}

int linux_file_close(File *f) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_close, lf->get_fd());
  if (ret < 0) {
    return -ret;
  }
  delete lf;
  return 0;
}

ErrorOr<File *> openfile(const char *path, const char *mode) {
  using ModeFlags = File::ModeFlags;
  auto modeflags = File::mode_flags(mode);
  if (modeflags == 0) {
    // return {nullptr, EINVAL};
    return Error(EINVAL);
  }
  long open_flags = 0;
  if (modeflags & ModeFlags(File::OpenMode::APPEND)) {
    open_flags = O_CREAT | O_APPEND;
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_WRONLY;
  } else if (modeflags & ModeFlags(File::OpenMode::WRITE)) {
    open_flags = O_CREAT | O_TRUNC;
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_WRONLY;
  } else {
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_RDONLY;
  }

  // File created will have 0666 permissions.
  constexpr long OPEN_MODE =
      S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

#ifdef SYS_open
  int fd =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_open, path, open_flags, OPEN_MODE);
#elif defined(SYS_openat)
  int fd = LIBC_NAMESPACE::syscall_impl<int>(SYS_openat, AT_FDCWD, path,
                                             open_flags, OPEN_MODE);
#else
#error "open and openat syscalls not available."
#endif

  if (fd < 0)
    return Error(-fd);

  uint8_t *buffer;
  {
    AllocChecker ac;
    buffer = new (ac) uint8_t[File::DEFAULT_BUFFER_SIZE];
    if (!ac)
      return Error(ENOMEM);
  }
  AllocChecker ac;
  auto *file = new (ac)
      LinuxFile(fd, buffer, File::DEFAULT_BUFFER_SIZE, _IOFBF, true, modeflags);
  if (!ac)
    return Error(ENOMEM);
  return file;
}

int get_fileno(File *f) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  return lf->get_fd();
}

} // namespace LIBC_NAMESPACE
