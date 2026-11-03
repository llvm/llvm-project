//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal helpers for flushing streams, shared by fflush and exit.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FFLUSH_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_FFLUSH_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<int> flush_stream(::FILE *stream);
ErrorOr<int> flush_all_streams();

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_FFLUSH_INTERNAL_H
