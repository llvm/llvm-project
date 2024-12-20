//===---------- UEFI implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "io.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

ssize_t read_from_stdin(char *buf, size_t size) {
  (void)buf;
  (void)size;
  return 0;
}

void write_to_stdout(cpp::string_view msg) { (void)msg; }

void write_to_stderr(cpp::string_view msg) { (void)msg; }

} // namespace LIBC_NAMESPACE_DECL
