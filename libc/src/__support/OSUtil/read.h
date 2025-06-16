//===--------- Implementation header of the read function -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_READ_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_READ_H

#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/unistd_macros.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<ssize_t> read(int fd, void *buf, size_t count);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_READ_H
