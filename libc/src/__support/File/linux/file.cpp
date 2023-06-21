//===--- Linux specialization of the File data structure ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"

#include "src/__support/CPP/new.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/errno/libc_errno.h"         // For error macros

#include <fcntl.h> // For mode_t and other flags to the open syscall
#include <stdio.h>
#include <sys/syscall.h> // For syscall numbers

namespace __llvm_libc {

namespace {

FileIOResult write_func(File *, const void *, size_t);
FileIOResult read_func(File *, void *, size_t);
ErrorOr<long> seek_func(File *, long, int);
int close_func(File *);

} // anonymous namespace

class LinuxFile : public File {
  int fd;

public:
  constexpr LinuxFile(int file_descriptor, uint8_t *buffer, size_t buffer_size,
                      int buffer_mode, bool owned, File::ModeFlags modeflags)
      : File(&write_func, &read_func, &seek_func, &close_func, buffer,
             buffer_size, buffer_mode, owned, modeflags),
        fd(file_descriptor) {}

  int get_fd() const { return fd; }
};

namespace {

FileIOResult write_func(File *f, const void *data, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret = __llvm_libc::syscall_impl(SYS_write, lf->get_fd(), data, size);
  if (ret < 0) {
    return {0, -ret};
  }
  return ret;
}

FileIOResult read_func(File *f, void *buf, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret = __llvm_libc::syscall_impl(SYS_read, lf->get_fd(), buf, size);
  if (ret < 0) {
    return {0, -ret};
  }
  return ret;
}

ErrorOr<long> seek_func(File *f, long offset, int whence) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  long result;
#ifdef SYS_lseek
  int ret = __llvm_libc::syscall_impl(SYS_lseek, lf->get_fd(), offset, whence);
  result = ret;
#elif defined(SYS__llseek)
  long result;
  int ret = __llvm_libc::syscall_impl(SYS__llseek, lf->get_fd(), offset >> 32,
                                      offset, &result, whence);
#else
#error "lseek and _llseek syscalls not available to perform a seek operation."
#endif

  if (ret < 0)
    return Error(-ret);

  return result;
}

int close_func(File *f) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  int ret = __llvm_libc::syscall_impl(SYS_close, lf->get_fd());
  if (ret < 0) {
    return -ret;
  }
  delete lf;
  return 0;
}

} // anonymous namespace

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
  int fd = __llvm_libc::syscall_impl(SYS_open, path, open_flags, OPEN_MODE);
#elif defined(SYS_openat)
  int fd = __llvm_libc::syscall_impl(SYS_openat, AT_FDCWD, path, open_flags,
                                     OPEN_MODE);
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

constexpr size_t STDIN_BUFFER_SIZE = 512;
uint8_t stdin_buffer[STDIN_BUFFER_SIZE];
static LinuxFile StdIn(0, stdin_buffer, STDIN_BUFFER_SIZE, _IOFBF, false,
                       File::ModeFlags(File::OpenMode::READ));
File *stdin = &StdIn;

constexpr size_t STDOUT_BUFFER_SIZE = 1024;
uint8_t stdout_buffer[STDOUT_BUFFER_SIZE];
static LinuxFile StdOut(1, stdout_buffer, STDOUT_BUFFER_SIZE, _IOLBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

constexpr size_t STDERR_BUFFER_SIZE = 0;
static LinuxFile StdErr(2, nullptr, STDERR_BUFFER_SIZE, _IONBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace __llvm_libc

// Provide the external defintitions of the standard IO streams.
extern "C" {
FILE *stdin = reinterpret_cast<FILE *>(&__llvm_libc::StdIn);
FILE *stderr = reinterpret_cast<FILE *>(&__llvm_libc::StdErr);
FILE *stdout = reinterpret_cast<FILE *>(&__llvm_libc::StdOut);
}
