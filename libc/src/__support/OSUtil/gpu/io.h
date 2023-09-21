//===-------------- GPU implementation of IO utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_GPU_IO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_GPU_IO_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE

namespace __llvm_libc {

void write_to_stderr(cpp::string_view msg);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_IO_H
