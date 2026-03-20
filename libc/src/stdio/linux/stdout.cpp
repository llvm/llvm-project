//===--- Definition of Linux stdout ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/stdout.h"

#include "hdr/types/FILE.h"

#ifdef LIBC_FULL_BUILD

#include "src/__support/File/linux/file.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constexpr size_t STDOUT_BUFFER_SIZE = 1024;
uint8_t stdout_buffer[STDOUT_BUFFER_SIZE];
static LinuxFile StdOut(1, stdout_buffer, STDOUT_BUFFER_SIZE, _IOLBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));

LLVM_LIBC_VARIABLE(FILE *, stdout) = reinterpret_cast<FILE *>(&StdOut);

} // namespace LIBC_NAMESPACE_DECL

#else // overlay mode

extern "C" FILE *stderr;

#endif // LIBC_FULL_BUILD
