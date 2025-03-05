//===-- Implementation of ftell -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/ftell.h"
#include "src/__support/File/file.h"

#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, ftell, (::FILE * stream)) {
  auto result = reinterpret_cast<LIBC_NAMESPACE::File *>(stream)->tell();
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  // tell() returns an off_t (64-bit signed integer), but this function returns
  // a long (32-bit signed integer in 32-bit systems). We add a cast here to
  // silence a "implicit conversion loses integer precision" warning when
  // compiling for 32-bit systems.
  return static_cast<long>(result.value());
}

} // namespace LIBC_NAMESPACE_DECL
