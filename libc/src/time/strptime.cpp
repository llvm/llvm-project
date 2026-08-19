//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the strptime function.
///
//===----------------------------------------------------------------------===//

#include "src/time/strptime.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, strptime,
                   ([[maybe_unused]] const char *__restrict buf,
                    [[maybe_unused]] const char *__restrict format,
                    [[maybe_unused]] const struct tm *__restrict tm)) {
  return nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
