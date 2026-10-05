//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for tcsetpgrp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_TCSETPGRP_H
#define LLVM_LIBC_SRC_UNISTD_TCSETPGRP_H

#include "hdr/types/pid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int tcsetpgrp(int fd, pid_t pgid);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_TCSETPGRP_H
