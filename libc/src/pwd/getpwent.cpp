//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getpwent.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/getpwent.h"
#include "src/__support/CPP/span.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/pwd/pwd_utils.h"

#include "hdr/stdio_macros.h"

#ifndef LIBC_COPT_PWD_FILE_PATH
#define LIBC_COPT_PWD_FILE_PATH "/etc/passwd"
#endif

namespace LIBC_NAMESPACE_DECL {

static File *pwd_file = nullptr;
static const char *pwd_file_path = LIBC_COPT_PWD_FILE_PATH;
// Note: These static buffers are process-global and NOT protected by a mutex
// at this stage. POSIX getpwent is non-reentrant.
static char line_buffer[1024];
static struct passwd pwd_entry;

namespace internal {
void set_passwd_path(const char *path) {
  if (!path)
    return;
  if (pwd_file) {
    pwd_file->close();
    pwd_file = nullptr;
  }
  pwd_file_path = path;
}
} // namespace internal

ErrorOr<int> setpwent_impl() {
  if (!pwd_file) {
    auto result = openfile(pwd_file_path, "r");
    if (!result.has_value())
      return Error(result.error());
    pwd_file = result.value();
  } else {
    auto result = pwd_file->seek(0, SEEK_SET);
    if (!result.has_value())
      return Error(result.error());
  }
  return 0;
}

ErrorOr<int> endpwent_impl() {
  if (pwd_file) {
    int result = pwd_file->close();
    pwd_file = nullptr;
    if (result != 0)
      return Error(result);
  }
  return 0;
}

struct ReadLineResult {
  size_t bytes_read;
  bool truncated;
};

// Reads a line from the given file into buf.
static ErrorOr<ReadLineResult> read_line(File *f, cpp::span<char> buf) {
  if (!f || buf.empty())
    return Error(EINVAL);

  f->lock();
  size_t bytes_read = 0;
  FileIOResult result(0);
  bool truncated = false;

  for (char &ch : buf.first(buf.size() - 1)) {
    result = f->read_unlocked(&ch, 1);
    if (result.has_error()) {
      f->unlock();
      return Error(result.error);
    }
    if (result.value != 1)
      break;
    ++bytes_read;
    if (ch == '\n')
      break;
  }

  if (result.value == 1 && bytes_read > 0 && buf[bytes_read - 1] != '\n') {
    truncated = true;
    char c = '\0';
    while (true) {
      result = f->read_unlocked(&c, 1);
      if (result.has_error()) {
        f->unlock();
        return Error(result.error);
      }
      if (result.value != 1 || c == '\n')
        break;
    }
  }

  bool has_error = f->error_unlocked();
  bool has_eof = f->iseof_unlocked();
  f->unlock();

  if (has_error)
    return Error(EIO);

  if (bytes_read == 0 && has_eof)
    return ReadLineResult{0, false};

  buf[bytes_read] = '\0';
  return ReadLineResult{bytes_read, truncated};
}

LLVM_LIBC_FUNCTION(struct passwd *, getpwent, ()) {
  if (!pwd_file) {
    auto result = openfile(pwd_file_path, "r");
    if (!result.has_value()) {
      libc_errno = result.error();
      return nullptr;
    }
    pwd_file = result.value();
  }

  while (true) {
    auto result = read_line(pwd_file, line_buffer);
    if (!result.has_value()) {
      libc_errno = result.error();
      return nullptr;
    }

    ReadLineResult res = result.value();
    if (res.bytes_read == 0)
      return nullptr;

    if (res.truncated) {
      libc_errno = EINVAL;
      return nullptr;
    }

    size_t len = res.bytes_read;
    if (len > 0 && line_buffer[len - 1] == '\n')
      line_buffer[len - 1] = '\0';

    auto passwd_or = internal::parse_passwd_line(line_buffer);
    if (!passwd_or.has_value()) {
      libc_errno = passwd_or.error();
      return nullptr;
    }

    pwd_entry = passwd_or.value();
    return &pwd_entry;
  }
}

} // namespace LIBC_NAMESPACE_DECL
