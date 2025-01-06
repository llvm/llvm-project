//===--- Definition of Linux stderr ---------------------------------------===//
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

constexpr size_t STDERR_BUFFER_SIZE = 0;
static LinuxFile StdErr(2, nullptr, STDERR_BUFFER_SIZE, _IONBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace LIBC_NAMESPACE_DECL

extern "C" {
FILE *stderr = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdErr);
}
