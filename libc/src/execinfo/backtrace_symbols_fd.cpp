//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of backtrace_symbols_fd
///
//===----------------------------------------------------------------------===//

#include "src/execinfo/backtrace_symbols_fd.h"

#include "hdr/types/dl_info.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/write.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dladdr.h"
#include "src/execinfo/backtrace_line.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

void write_all(int fd, const char *text, size_t length) {
  size_t done = 0;
  while (done < length) {
    const ErrorOr<ssize_t> written =
        linux_syscalls::write(fd, text + done, length - done);
    if (!written || *written <= 0)
      return;
    done += static_cast<size_t>(*written);
  }
}

} // anonymous namespace

// One line per address, in the form the other implementations use:
//
//   /path/to/object(symbol+0x1c) [0x7f0e2c001234]
//
// with whichever parts are known. Nothing is allocated, so unlike
// backtrace_symbols this can be called from a signal handler.
LLVM_LIBC_FUNCTION(void, backtrace_symbols_fd,
                   (void *const *buffer, int size, int fd)) {
  if (buffer == nullptr)
    return;

  for (int i = 0; i < size; ++i) {
    char line[execinfo::MAX_LINE];
    char *out = line;
    char *const end = line + sizeof(line) - 1;

    Dl_info info = {};
    const bool found = internal::dladdr_impl(buffer[i], &info) != 0;
    if (found && info.dli_fname != nullptr)
      out = execinfo::put(out, end, info.dli_fname);
    if (found && info.dli_sname != nullptr) {
      out = execinfo::put(out, end, '(');
      out = execinfo::put(out, end, info.dli_sname);
      const uintptr_t offset = reinterpret_cast<uintptr_t>(buffer[i]) -
                               reinterpret_cast<uintptr_t>(info.dli_saddr);
      out = execinfo::put(out, end, '+');
      out = execinfo::put_hex(out, end, offset);
      out = execinfo::put(out, end, ')');
    }
    if (out != line)
      out = execinfo::put(out, end, ' ');
    out = execinfo::put(out, end, '[');
    out = execinfo::put_hex(out, end, reinterpret_cast<uintptr_t>(buffer[i]));
    out = execinfo::put(out, end, ']');
    out = execinfo::put(out, end, '\n');

    write_all(fd, line, static_cast<size_t>(out - line));
  }
}

} // namespace LIBC_NAMESPACE_DECL
