//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper functions for pwd.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/pwd_utils.h"
#include "hdr/errno_macros.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/struct_passwd.h"
#include "src/__support/CPP/span.h"
#include "src/__support/File/file.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"
#include "src/string/string_utils.h"

#ifndef LIBC_COPT_PWD_FILE_PATH
#define LIBC_COPT_PWD_FILE_PATH "/etc/passwd"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<struct passwd> parse_passwd_line(char *line) {
  if (!line)
    return Error(EINVAL);

  struct passwd pwd;
  char *context = line;

  pwd.pw_name = string_token<false>(nullptr, ":", &context);
  if (!pwd.pw_name)
    return Error(EINVAL);

  pwd.pw_passwd = string_token<false>(nullptr, ":", &context);
  if (!pwd.pw_passwd)
    return Error(EINVAL);

  char *uid_str = string_token<false>(nullptr, ":", &context);
  if (!uid_str || !isdigit(uid_str[0]))
    return Error(EINVAL);
  auto uid_res = strtointeger<uid_t>(uid_str, 10);
  if (uid_res.has_error() || uid_res.parsed_len == 0 ||
      uid_str[uid_res.parsed_len] != '\0')
    return Error(EINVAL);
  pwd.pw_uid = uid_res.value;

  char *gid_str = string_token<false>(nullptr, ":", &context);
  if (!gid_str || !isdigit(gid_str[0]))
    return Error(EINVAL);
  auto gid_res = strtointeger<gid_t>(gid_str, 10);
  if (gid_res.has_error() || gid_res.parsed_len == 0 ||
      gid_str[gid_res.parsed_len] != '\0')
    return Error(EINVAL);
  pwd.pw_gid = gid_res.value;

  pwd.pw_gecos = string_token<false>(nullptr, ":", &context);
  if (!pwd.pw_gecos)
    return Error(EINVAL);

  pwd.pw_dir = string_token<false>(nullptr, ":", &context);
  if (!pwd.pw_dir)
    return Error(EINVAL);

  pwd.pw_shell = string_token<false>(nullptr, ":", &context);
  if (!pwd.pw_shell)
    return Error(EINVAL);

  return pwd;
}

} // namespace internal

namespace passwd {

static File *pwd_file = nullptr;
static const char *pwd_file_path = LIBC_COPT_PWD_FILE_PATH;
// Note: These static buffers are process-global and NOT protected by a mutex
// at this stage. POSIX getpwent is non-reentrant.
static char line_buffer[1024];
static struct passwd pwd_entry;

void TESTONLY_set_passwd_path(const char *path) {
  if (!path)
    return;
  if (pwd_file) {
    pwd_file->close();
    pwd_file = nullptr;
  }
  pwd_file_path = path;
}

ErrorOr<void> open() {
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
  return {};
}

ErrorOr<void> close() {
  if (pwd_file) {
    int result = pwd_file->close();
    pwd_file = nullptr;
    if (result != 0)
      return Error(result);
  }
  return {};
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
  f->unlock();

  if (has_error)
    return Error(EIO);

  buf[bytes_read] = '\0';
  return ReadLineResult{bytes_read, truncated};
}

ErrorOr<struct passwd *> read_next() {
  if (!pwd_file) {
    auto result = open();
    if (!result.has_value())
      return Error(result.error());
  }

  while (true) {
    auto result = read_line(pwd_file, line_buffer);
    if (!result.has_value())
      return Error(result.error());

    ReadLineResult res = result.value();
    if (res.bytes_read == 0)
      return nullptr;

    if (res.truncated)
      return Error(EINVAL);

    size_t len = res.bytes_read;
    if (len > 0 && line_buffer[len - 1] == '\n')
      line_buffer[len - 1] = '\0';

    auto passwd_or = internal::parse_passwd_line(line_buffer);
    if (!passwd_or.has_value())
      return Error(passwd_or.error());

    pwd_entry = passwd_or.value();
    return &pwd_entry;
  }
}

} // namespace passwd
} // namespace LIBC_NAMESPACE_DECL
