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
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"
#include "src/string/string_utils.h"

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
} // namespace LIBC_NAMESPACE_DECL
