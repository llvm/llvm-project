//===-- Implementation of catgets -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/nl_types/catgets.h"
#include "include/llvm-libc-types/nl_catd.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, catgets,
                   ([[maybe_unused]] nl_catd catalog,
                    [[maybe_unused]] int set_number,
                    [[maybe_unused]] int message_number, const char *message)) {
  // TODO: Add implementation for message catalogs. For now, return backup
  // message regardless of input.
  return const_cast<char *>(message);
}

} // namespace LIBC_NAMESPACE_DECL
