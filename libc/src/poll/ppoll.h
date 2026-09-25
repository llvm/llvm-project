//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for ppoll.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_POLL_PPOLL_H
#define LLVM_LIBC_SRC_POLL_PPOLL_H

#include "hdr/types/nfds_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_pollfd.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int ppoll(struct pollfd *fds, nfds_t nfds,
          const struct timespec *__restrict tmo_p,
          const sigset_t *__restrict sigmask);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_POLL_PPOLL_H
