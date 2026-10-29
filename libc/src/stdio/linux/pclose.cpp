//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the POSIX pclose function for Linux.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/pclose.h"

#include "hdr/types/FILE.h"
#include "hdr/types/pid_t.h"
#include "src/__support/File/linux/popen_file.h"
#include "src/__support/OSUtil/linux/shell_fork.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pclose, (::FILE * stream)) {
  LIBC_CRASH_ON_NULLPTR(stream);

  // Passing a stream that was not created by popen is undefined behavior.
  auto *file = reinterpret_cast<PopenFile *>(stream);
  const pid_t pid = file->get_pid();

  // Closing the stream flushes it and closes the parent's end of the pipe, so
  // a child reading from it sees EOF. It also frees the stream.
  // TODO: File::close returns early without closing the fd if flushing fails,
  // which could leave a child reading from the pipe waiting forever. Fix
  // File::close to always release the fd in a followup.
  int close_result = file->close();

  // POSIX requires that pclose not return before the child has terminated, so
  // wait even if closing failed.
  ErrorOr<int> wait_result = shell_wait(pid);
  if (!wait_result) {
    libc_errno = wait_result.error();
    return -1;
  }
  if (close_result != 0) {
    libc_errno = close_result;
    return -1;
  }
  return wait_result.value();
}

} // namespace LIBC_NAMESPACE_DECL
