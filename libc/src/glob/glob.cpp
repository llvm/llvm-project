//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of glob.
///
//===----------------------------------------------------------------------===//

#include "src/glob/glob.h"
#include "hdr/glob_macros.h"
#include "hdr/types/glob_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, glob,
                   ([[maybe_unused]] const char *__restrict pattern,
                    [[maybe_unused]] int flags,
                    [[maybe_unused]] int (*errfunc)(const char *, int),
                    [[maybe_unused]] glob_t *__restrict pglob)) {
  // TODO: Implement glob.
  return GLOB_NOMATCH;
}

} // namespace LIBC_NAMESPACE_DECL
