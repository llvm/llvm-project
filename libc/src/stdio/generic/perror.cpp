//===-- Implementation of perror ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/perror.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/StringUtil/error_to_string.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

static int write_out(cpp::string_view str_view, File *f) {
  if (str_view.size() > 0) {
    auto result = f->write_unlocked(str_view.data(), str_view.size());
    if (result.has_error())
      return result.error;
  }
  return 0;
}

// separate function so that we can return early on error but still get the
// unlock. This function sets errno and should not be called elsewhere.
static void write_sequence(cpp::string_view str_view,
                           cpp::string_view err_str) {
  int write_err;
  // TODO: this seems like there should be some sort of queue system to
  // deduplicate this code.

  // FORMAT:
  // if str != nullptr and doesn't start with a null byte:
  //   "[str]: [strerror(errno)]\n"
  // else
  //   "[strerror(errno)]\n"
  if (str_view.size() > 0) {
    write_err = write_out(str_view, LIBC_NAMESPACE::stderr);
    if (write_err != 0) {
      libc_errno = write_err;
      return;
    }

    write_err = write_out(": ", LIBC_NAMESPACE::stderr);
    if (write_err != 0) {
      libc_errno = write_err;
      return;
    }
  }

  write_err = write_out(err_str, LIBC_NAMESPACE::stderr);
  if (write_err != 0) {
    libc_errno = write_err;
    return;
  }

  write_err = write_out("\n", LIBC_NAMESPACE::stderr);
  if (write_err != 0) {
    libc_errno = write_err;
    return;
  }
}

LLVM_LIBC_FUNCTION(void, perror, (const char *str)) {
  const char empty_str[1] = {'\0'};
  if (str == nullptr)
    str = empty_str;
  cpp::string_view str_view(str);

  cpp::string_view err_str = get_error_string(libc_errno);

  // We need to lock the stream to ensure the newline is always appended.
  LIBC_NAMESPACE::stderr->lock();
  write_sequence(str_view, err_str);
  LIBC_NAMESPACE::stderr->unlock();
}

} // namespace LIBC_NAMESPACE_DECL
