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
#include "src/__support/str_to_integer.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

bool parse_passwd_line(char *line, struct passwd *pwd) {
  if (!line || !pwd)
    return false;

  char *context = line;

  pwd->pw_name =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!pwd->pw_name)
    return false;

  pwd->pw_passwd =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!pwd->pw_passwd)
    return false;

  char *uid_str =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!uid_str)
    return false;
  auto uid_res = LIBC_NAMESPACE::internal::strtointeger<uid_t>(uid_str, 10);
  if (uid_res.has_error())
    return false;
  pwd->pw_uid = uid_res;

  char *gid_str =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!gid_str)
    return false;
  auto gid_res = LIBC_NAMESPACE::internal::strtointeger<gid_t>(gid_str, 10);
  if (gid_res.has_error())
    return false;
  pwd->pw_gid = gid_res;

  pwd->pw_gecos =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!pwd->pw_gecos)
    return false;

  pwd->pw_dir =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!pwd->pw_dir)
    return false;

  // shell
  pwd->pw_shell =
      LIBC_NAMESPACE::internal::string_token<false>(nullptr, ":", &context);
  if (!pwd->pw_shell)
    return false;

  return true;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
