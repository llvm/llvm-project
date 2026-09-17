//===--- Helpers for file I/O on baremetal ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/baremetal/file_internal.h"
#include "src/__support/CPP/optional.h"

#include "hdr/stdio_macros.h" // for EOF, FILE

namespace LIBC_NAMESPACE_DECL {

// The fallback handles stdin only because it is the only readable standard
// stream. The C standard requires only one byte of pushback. Applications that
// need pushback for other streams or integrate it with their input buffering
// can provide __llvm_libc_stdio_ungetc instead.

static cpp::optional<unsigned char> ungetc_state_stdin;

bool pop_ungetc_value(::FILE *stream, unsigned char &out) {
  if (stream != stdin)
    return false;

  if (!ungetc_state_stdin)
    return false;

  out = *ungetc_state_stdin;
  ungetc_state_stdin.reset();
  return true;
}

int push_ungetc_value(::FILE *stream, int c) {
  if (c == EOF || stream == nullptr)
    return EOF;

  if (__llvm_libc_stdio_ungetc != nullptr)
    return __llvm_libc_stdio_ungetc(stream, c);

  if (stream != stdin)
    return EOF;

  if (ungetc_state_stdin)
    return EOF;

  ungetc_state_stdin =
      cpp::optional<unsigned char>{static_cast<unsigned char>(c)};
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
