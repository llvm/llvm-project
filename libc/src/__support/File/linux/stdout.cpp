//===--- Definition of Linux stdout ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/linux/file.h"
#include "src/__support/common.h"

#include <stdio.h>

namespace LIBC_NAMESPACE {

constexpr size_t STDOUT_BUFFER_SIZE = 1024;
uint8_t stdout_buffer[STDOUT_BUFFER_SIZE];
static LinuxFile StdOut(1, stdout_buffer, STDOUT_BUFFER_SIZE, _IOLBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

} // namespace LIBC_NAMESPACE

LLVM_LIBC_GLOBAL(FILE *,
                 stdout) = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdOut);
