//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for eventfd_read.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_EVENTFD_EVENTFD_READ_H
#define LLVM_LIBC_SRC_SYS_EVENTFD_EVENTFD_READ_H

#include "hdr/types/eventfd_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int eventfd_read(int fd, eventfd_t *value);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_EVENTFD_EVENTFD_READ_H
