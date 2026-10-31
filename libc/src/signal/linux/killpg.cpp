//===-- Linux implementation of killpg ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/killpg.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/signal/kill.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, killpg, (pid_t pgrp, int sig)) {
    if(pgrp < 0) {
        libc_errno = EINVAL;
        return -1;
    }
    return LIBC_NAMESPACE::kill(-pgrp, sig);
 } 
} //namespace LIBC_NAMESPACE_DECL