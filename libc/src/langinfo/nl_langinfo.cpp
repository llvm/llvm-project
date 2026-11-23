//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of nl_langinfo.
///
//===----------------------------------------------------------------------===//

#include "src/langinfo/nl_langinfo.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/langinfo/langinfo_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, nl_langinfo, (nl_item item)) {
  // POSIX requires a char * return type, though callers may not modify it.
  return const_cast<char *>(get_default_langinfo_string(item));
}

} // namespace LIBC_NAMESPACE_DECL
