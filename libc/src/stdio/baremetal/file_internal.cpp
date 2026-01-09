//===--- Helpers for file I/O on baremetal ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/baremetal/file_internal.h"

#include "hdr/stdio_macros.h" // for EOF

namespace LIBC_NAMESPACE_DECL {

// Baremetal only exposes three fixed streams now and only stdin supports ungetc
// because stdin is readable - ungetc on stdout/stderr is undefined.
// Only one value is required by the C standard to be stored by ungetc.
extern "C" ::FILE *stdin;

struct UngetcState {
  bool has_value;
  unsigned char value;
};

static UngetcState ungetc_state_stdin{false, 0};

bool pop_ungetc_value(::FILE *stream, unsigned char &out) {
  if (stream != stdin)
    return false;

  if (ungetc_state_stdin.has_value) {
    out = ungetc_state_stdin.value;
    ungetc_state_stdin.has_value = false;
    return true;
  }
  return false;
}

int store_ungetc_value(::FILE *stream, int c) {
  if (c == EOF || stream == nullptr)
    return EOF;

  if (stream != stdin)
    return EOF;

  if (ungetc_state_stdin.has_value)
    return EOF;

  ungetc_state_stdin.value = static_cast<unsigned char>(c);
  ungetc_state_stdin.has_value = true;
  return ungetc_state_stdin.value;
}

} // namespace LIBC_NAMESPACE_DECL
