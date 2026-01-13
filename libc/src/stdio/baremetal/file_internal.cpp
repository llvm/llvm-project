//===--- Helpers for file I/O on baremetal ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/baremetal/file_internal.h"
#include "src/__support/CPP/optional.h"

#include "hdr/stdio_macros.h" // for EOF

namespace LIBC_NAMESPACE_DECL {

// Out of standard streams only stdin supports ungetc,
// because stdin is readable - ungetc on stdout/stderr is undefined.
// Only one value is required by the C standard to be stored by ungetc.
// This minimal implementation only handles stdin and returns error on all
// other streams.
// TODO: Shall we have an embedding API for ungetc?
extern "C" ::FILE *stdin;

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

int store_ungetc_value(::FILE *stream, int c) {
  if (c == EOF || stream == nullptr)
    return EOF;

  if (stream != stdin)
    return EOF;

  if (ungetc_state_stdin)
    return EOF;

  ungetc_state_stdin =
      cpp::optional<unsigned char>{static_cast<unsigned char>(c)};
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
