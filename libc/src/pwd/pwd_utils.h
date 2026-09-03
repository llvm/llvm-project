//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of helper functions and parser for pwd.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PWD_PWD_UTILS_H
#define LLVM_LIBC_SRC_PWD_PWD_UTILS_H

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "src/__support/CPP/span.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_integer.h"
#include "src/pwd/field_tokenizer.h"
#include "src/pwd/flat_file_db.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

// Parses a colon-separated line in-place into a struct passwd.
template <>
LIBC_INLINE bool parse_line<struct passwd>(cpp::span<char> line,
                                           struct passwd *pwd) {
  if (line.empty() || !pwd)
    return false;

  FieldTokenizer tokenizer(line);

  auto name = tokenizer.next_field();
  if (!name)
    return false;
  pwd->pw_name = name->data();

  auto passwd = tokenizer.next_field();
  if (!passwd)
    return false;
  pwd->pw_passwd = passwd->data();

  auto uid_str = tokenizer.next_field();
  if (!uid_str || uid_str->empty() || !internal::isdigit(uid_str->front()))
    return false;
  auto uid_res = internal::strtointeger<uid_t>(uid_str->data(), 10);
  if (uid_res.has_error() || uid_res.parsed_len <= 0 ||
      static_cast<size_t>(uid_res.parsed_len) >= uid_str->size() ||
      (*uid_str)[uid_res.parsed_len] != '\0')
    return false;
  pwd->pw_uid = uid_res.value;

  auto gid_str = tokenizer.next_field();
  if (!gid_str || gid_str->empty() || !internal::isdigit(gid_str->front()))
    return false;
  auto gid_res = internal::strtointeger<gid_t>(gid_str->data(), 10);
  if (gid_res.has_error() || gid_res.parsed_len <= 0 ||
      static_cast<size_t>(gid_res.parsed_len) >= gid_str->size() ||
      (*gid_str)[gid_res.parsed_len] != '\0')
    return false;
  pwd->pw_gid = gid_res.value;

  auto gecos = tokenizer.next_field();
  if (!gecos)
    return false;
  pwd->pw_gecos = gecos->data();

  auto dir = tokenizer.next_field();
  if (!dir)
    return false;
  pwd->pw_dir = dir->data();

  auto shell = tokenizer.next_field();
  if (!shell)
    return false;
  pwd->pw_shell = shell->data();

  return true;
}

// Parses a colon-separated password database line into a struct passwd.
ErrorOr<struct passwd> parse_passwd_line(char *line);

} // namespace pwd

namespace passwd {

// Overrides the default password file path for testing purposes.
void TESTONLY_set_passwd_path(const char *path);

// Opens or rewinds the password file.
ErrorOr<void> open();

// Closes the password file.
ErrorOr<void> close();

// Reads the next entry from the password database.
ErrorOr<struct passwd *> read_next();

} // namespace passwd
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PWD_PWD_UTILS_H
