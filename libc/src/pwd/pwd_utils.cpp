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
#include "hdr/types/struct_passwd.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/attributes.h"
#include "src/pwd/flat_file_db.h"
#include "src/string/string_utils.h"

#ifndef LIBC_COPT_PWD_FILE_PATH
#define LIBC_COPT_PWD_FILE_PATH "/etc/passwd"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

ErrorOr<struct passwd> parse_passwd_line(char *line) {
  if (!line)
    return Error(EINVAL);

  struct passwd pwd;
  size_t len = internal::string_length(line);
  if (!parse_line(cpp::span<char>(line, len + 1), &pwd))
    return Error(EINVAL);

  return pwd;
}

// Exposed via TESTONLY_set_passwd_path for unit testing to direct operations
// to hermetic temporary files.
static const char *passwd_file_path = LIBC_COPT_PWD_FILE_PATH;

static LIBC_CONSTINIT FlatFileDatabase<struct passwd>
    db(LIBC_COPT_PWD_FILE_PATH);
// Note: These static buffers are process-global and NOT protected by a mutex
// at this stage. POSIX getpwent is non-reentrant.
static char line_buffer[1024];
static struct passwd pwd_entry;

void TESTONLY_set_passwd_path(const char *path) {
  passwd_file_path = path;
  db.set_path(path);
}

void TESTONLY_reset_passwd_path() {
  passwd_file_path = LIBC_COPT_PWD_FILE_PATH;
  db.set_path(LIBC_COPT_PWD_FILE_PATH);
}

ErrorOr<void> open() { return db.setdb(); }

ErrorOr<void> close() { return db.enddb(); }

ErrorOr<struct passwd *> read_next() {
  auto res = db.getnext(&pwd_entry, line_buffer);
  if (!res.has_value())
    return Error(res.error());
  if (!res.value())
    return nullptr;
  return &pwd_entry;
}

ErrorOr<bool> find_by_name(cpp::string_view name, struct passwd *pwd,
                           cpp::span<char> buffer, const char *path) {
  ScopedFlatFileDatabase<struct passwd> local_db(path ? path
                                                      : passwd_file_path);
  auto matcher = [name](const struct passwd &entry) {
    return cpp::string_view(entry.pw_name) == name;
  };
  return local_db.lookup(matcher, pwd, buffer);
}

ErrorOr<bool> find_by_uid(uid_t uid, struct passwd *pwd, cpp::span<char> buffer,
                          const char *path) {
  ScopedFlatFileDatabase<struct passwd> local_db(path ? path
                                                      : passwd_file_path);
  auto matcher = [uid](const struct passwd &entry) {
    return entry.pw_uid == uid;
  };
  return local_db.lookup(matcher, pwd, buffer);
}

ErrorOr<struct passwd *> find_by_name(cpp::string_view name) {
  auto res = find_by_name(name, &pwd_entry, line_buffer);
  if (!res.has_value())
    return Error(res.error());
  bool found = res.value();
  if (!found)
    return nullptr;
  return &pwd_entry;
}

ErrorOr<struct passwd *> find_by_uid(uid_t uid) {
  auto res = find_by_uid(uid, &pwd_entry, line_buffer);
  if (!res.has_value())
    return Error(res.error());
  bool found = res.value();
  if (!found)
    return nullptr;
  return &pwd_entry;
}

} // namespace pwd
} // namespace LIBC_NAMESPACE_DECL
