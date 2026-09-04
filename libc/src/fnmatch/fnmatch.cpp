//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the fnmatch function.
///
//===----------------------------------------------------------------------===//

#include "src/fnmatch/fnmatch.h"
#include "hdr/fnmatch_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fnmatch,
                   ([[maybe_unused]] const char *pattern,
                    [[maybe_unused]] const char *string,
                    [[maybe_unused]] int flags)) {
  // Always return FNM_NOMATCH for now.
  return FNM_NOMATCH;
}

} // namespace LIBC_NAMESPACE_DECL
