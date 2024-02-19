//===-- Header file of do_start -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/app.h"

namespace LIBC_NAMESPACE {
// setup the libc runtime and invoke the main routine.
[[noreturn]] void do_start();
} // namespace LIBC_NAMESPACE
