//===---------- Baremetal implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io.h"

#include "src/__support/CPP/string_view.h"

namespace LIBC_NAMESPACE {

// This is intended to be provided by the vendor.

extern struct __llvm_libc_stdin __llvm_libc_stdin;
extern "C" ssize_t __llvm_libc_stdin_read(void *cookie, char *buf, size_t size);

extern "C" void __llvm_libc_log_write(const char *msg, size_t len);

ssize_t read_from_stdin(char *buf, size_t size) {
  return __llvm_libc_stdin_read(reinterpret_cast<void *>(&__llvm_libc_stdin),
                                buf, size);
}

void write_to_stderr(cpp::string_view msg) {
  __llvm_libc_log_write(msg.data(), msg.size());
}

} // namespace LIBC_NAMESPACE
