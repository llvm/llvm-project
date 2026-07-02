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
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/pwd/pwd_utils.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fopen.h"
#include "src/string/string_utils.h"

#include "hdr/stdio_macros.h"
#include "src/stdio/fseek.h"

namespace LIBC_NAMESPACE_DECL {

static FILE *pwd_file = nullptr;
static char line_buffer[1024];
static struct passwd pwd_entry;

void setpwent_impl() {
  if (pwd_file)
    LIBC_NAMESPACE::fseek(pwd_file, 0, SEEK_SET);
}

void endpwent_impl() {
  if (pwd_file) {
    LIBC_NAMESPACE::fclose(pwd_file);
    pwd_file = nullptr;
  }
}

LLVM_LIBC_FUNCTION(struct passwd *, getpwent, ()) {
  if (!pwd_file) {
    pwd_file = LIBC_NAMESPACE::fopen("/etc/passwd", "r");
    if (!pwd_file)
      return nullptr;
  }

  while (LIBC_NAMESPACE::fgets(line_buffer, sizeof(line_buffer), pwd_file)) {
    // Remove newline
    size_t len = LIBC_NAMESPACE::internal::string_length(line_buffer);
    if (len > 0 && line_buffer[len - 1] == '\n')
      line_buffer[len - 1] = '\0';

    if (internal::parse_passwd_line(line_buffer, &pwd_entry))
      return &pwd_entry;
  }

  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
