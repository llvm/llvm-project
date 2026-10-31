//===-- Implementation header for killpg function ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_KILLPG_H
#define LLVM_LIBC_SRC_SIGNAL_KILLPG_H

#include "hdr/types/pid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int killpg(pid_t pgrp, int sig);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SIGNAL_KILLPG_H