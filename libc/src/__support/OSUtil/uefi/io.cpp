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
#include <Uefi.h>

namespace LIBC_NAMESPACE_DECL {

ssize_t read_from_stdin(char *, size_t) { return 0; }

void write_to_stdout(cpp::string_view msg) {
  // TODO: use mbstowcs once implemented
  for (size_t i = 0; i < msg.size(); i++) {
    char16_t e[2] = {msg[i], 0};
    efi_system_table->ConOut->OutputString(
        efi_system_table->ConOut, reinterpret_cast<const char16_t *>(&e));
  }
}

void write_to_stderr(cpp::string_view msg) {
  // TODO: use mbstowcs once implemented
  for (size_t i = 0; i < msg.size(); i++) {
    char16_t e[2] = {msg[i], 0};
    efi_system_table->StdErr->OutputString(
        efi_system_table->StdErr, reinterpret_cast<const char16_t *>(&e));
  }
}

} // namespace LIBC_NAMESPACE_DECL
