//===---------- Baremetal implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_IO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_IO_H

#include "hdr/types/off_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// These are intended to be provided by the vendor.
//
// The signature of these types and functions intentionally match `fopencookie`
// which allows the following:
//
// ```
// struct __llvm_libc_stdio_cookie { ... };
// ...
// struct __llvm_libc_stdio_cookie __llvm_libc_stdin_cookie;
// cookie_io_functions_t stdin_func = { .read = __llvm_libc_stdio_read };
// FILE *stdin = fopencookie(&__llvm_libc_stdin_cookie, "r", stdin_func);
// ...
// struct __llvm_libc_stdio_cookie __llvm_libc_stdout_cookie;
// cookie_io_functions_t stdout_func = { .write = __llvm_libc_stdio_write };
// FILE *stdout = fopencookie(&__llvm_libc_stdout_cookie, "w", stdout_func);
// ...
// struct __llvm_libc_stdio_cookie __llvm_libc_stderr_cookie;
// cookie_io_functions_t stderr_func = { .write = __llvm_libc_stdio_write };
// FILE *stderr = fopencookie(&__llvm_libc_stderr_cookie, "w", stderr_func);
// ```
//
// At the same time, implementation of functions like `printf` and `scanf` can
// use `__llvm_libc_stdio_read` and `__llvm_libc_stdio_write` directly to avoid
// the extra indirection.
//
// All three symbols `__llvm_libc_stdin_cookie`, `__llvm_libc_stdout_cookie`,
// and `__llvm_libc_stderr_cookie` must be provided, even if they don't point
// at anything.

struct __llvm_libc_stdio_cookie;

// On success, store a non-null application-owned cookie in `cookie` and return
// 0. On failure, return a negative errno value.
extern "C" int __llvm_libc_stdio_open(const char *path, const char *mode,
                                      void **cookie);

// Remove the file named by `path`. Return 0 on success or a negative errno
// value on failure.
extern "C" int __llvm_libc_stdio_remove(const char *path);

// Rename the file named by `old_path` to `new_path`. Return 0 on success or a
// negative errno value on failure.
extern "C" int __llvm_libc_stdio_rename(const char *old_path,
                                        const char *new_path);

// Return the number of bytes read, which can be less than `size` and is zero at
// end-of-file. On failure, return a negative errno value.
extern "C" ssize_t __llvm_libc_stdio_read(void *cookie, char *buf, size_t size);

// Return the number of bytes written, which can be less than `size`. On
// failure, return a negative errno value.
extern "C" ssize_t __llvm_libc_stdio_write(void *cookie, const char *buf,
                                           size_t size);

// Return the resulting absolute file position on success. On failure, return a
// negative errno value.
extern "C" off_t __llvm_libc_stdio_seek(void *cookie, off_t offset, int whence);

// Configure buffering for `cookie`. Return 0 on success or nonzero on failure,
// matching setvbuf. The application owns any supplied buffer.
extern "C" int __llvm_libc_stdio_set_buffer(void *cookie, char *buffer,
                                            size_t size, int mode);

// Flush buffered output for `cookie`, or all output streams if `cookie` is
// null. Return 0 on success or EOF on failure, matching fflush.
extern "C" int __llvm_libc_stdio_flush(void *cookie);

// Push `c` back onto the input stream for `cookie`. Return `c` on success or
// EOF on failure, matching ungetc. This hook is optional; when it is absent,
// LLVM libc provides one byte of pushback for stdin.
extern "C" [[gnu::weak]] int __llvm_libc_stdio_ungetc(void *cookie, int c);

// Return 0 on success or EOF on failure, matching fclose.
extern "C" int __llvm_libc_stdio_close(void *cookie);

void write_to_stderr(cpp::string_view msg);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_IO_H
