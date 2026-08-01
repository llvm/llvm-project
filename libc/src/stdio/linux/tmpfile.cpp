//===-- Implementation of tmpfile --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/tmpfile.h"
#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "hdr/types/FILE.h"

#include "src/__support/File/linux/file.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constexpr char TMPDIR[] = "/tmp";

LLVM_LIBC_FUNCTION(::FILE *, tmpfile, (void)) {
  auto fd = linux_syscalls::open(TMPDIR, O_RDWR | O_TMPFILE | O_EXCL, 0600);
  if (!fd.has_value()) {
    libc_errno = fd.error();
    return nullptr;
  }
  auto file = LIBC_NAMESPACE::create_file_from_fd(fd.value(), "w+b");
  if (!file.has_value()) {
    libc_errno = file.error();
    return nullptr;
  }
  return reinterpret_cast<::FILE *>(file.value());
}

} // namespace LIBC_NAMESPACE_DECL
