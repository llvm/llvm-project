//===-- Implementation of puts --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"

#include "hdr/types/FILE.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

namespace {

// Simple helper to unlock the file once destroyed.
struct ScopedLock {
  ScopedLock(LIBC_NAMESPACE::File *stream) : stream(stream) { stream->lock(); }
  ~ScopedLock() { stream->unlock(); }

private:
  LIBC_NAMESPACE::File *stream;
};

} // namespace

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  cpp::string_view str_view(str);

  // We need to lock the stream to ensure the newline is always appended.
  ScopedLock lock(LIBC_NAMESPACE::stdout);

  auto result = LIBC_NAMESPACE::stdout->write_unlocked(str, str_view.size());
  if (result.has_error())
    libc_errno = result.error;
  size_t written = result.value;
  if (str_view.size() != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  result = LIBC_NAMESPACE::stdout->write_unlocked("\n", 1);
  if (result.has_error())
    libc_errno = result.error;
  written = result.value;
  if (1 != written) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
