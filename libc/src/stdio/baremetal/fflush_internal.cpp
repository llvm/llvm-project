//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Baremetal implementation of the internal flushing helpers.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush_internal.h"

#include "hdr/types/FILE.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Baremetal uses unbuffered I/O, so there is nothing to flush.
// TODO: Shall we have an embedding API for fflush?

ErrorOr<int> flush_stream(::FILE *stream) {
  (void)stream;
  return 0;
}

ErrorOr<int> flush_all_streams() { return 0; }

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
