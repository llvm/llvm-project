//===--- Definition of baremetal stdout -----------------------------------===//
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

cookie_io_functions_t io_func = {.read = nullptr,
                                 .write = __llvm_libc_stdio_write,
                                 .seek = nullptr,
                                 .close = nullptr};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
// Buffering is implementation defined. Therefore to save RAM, we use no
// buffering
static CookieFile StdOut(&__llvm_libc_stdout_cookie, io_func, nullptr, 0,
                         _IONBF, File::mode_flags("w"));
#pragma clang diagnostic pop
File *stdout = &StdOut;

} // namespace LIBC_NAMESPACE_DECL

extern "C" {
FILE *stdout = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::StdOut);
}
