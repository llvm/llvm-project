//===--- Definition of Linux stdout ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constexpr size_t STDOUT_BUFFER_SIZE = 1024;
uint8_t stdout_buffer[STDOUT_BUFFER_SIZE];
static LinuxFile StdOut(1, stdout_buffer, STDOUT_BUFFER_SIZE, _IOLBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

} // namespace LIBC_NAMESPACE_DECL

extern "C" {
FILE *stdout = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdOut);
} // extern "C"
