//===--- Definition of baremetal stderr -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "src/__support/File/cookie_file.h"
#include "src/__support/File/file.h"
#include "src/__support/OSUtil/baremetal/io.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// To save space, all streams are aliased to one stream. Furthermore, no
// buffering is used.
cookie_io_functions_t io_func = {.read = __llvm_libc_stdio_read,
                                 .write = __llvm_libc_stdio_write,
                                 .seek = nullptr,
                                 .close = nullptr};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
static CookieFile StdErr(&__llvm_libc_stderr_cookie, io_func, nullptr, 0,
                         _IONBF, File::mode_flags("r+"));
#pragma clang diagnostic pop
File *stderr = &StdErr;
File *stdin = &StdErr;
File *stdout = &StdErr;

} // namespace LIBC_NAMESPACE_DECL

extern "C" {
FILE *stderr = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdErr);
FILE *stdin = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdErr);
FILE *stdout = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdErr);
}
