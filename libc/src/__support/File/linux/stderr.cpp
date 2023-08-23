//===--- Definition of Linux stderr ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"
#include <stdio.h>

namespace __llvm_libc {

constexpr size_t STDERR_BUFFER_SIZE = 0;
static LinuxFile StdErr(2, nullptr, STDERR_BUFFER_SIZE, _IONBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace __llvm_libc

extern "C" {
FILE *stderr = reinterpret_cast<FILE *>(&__llvm_libc::StdErr);
}
