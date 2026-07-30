//===-- Linux implementation of remove ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/remove.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: This is a temporary workaround for issue #85335.

LLVM_LIBC_FUNCTION(int, remove, (const char *)) { return -1; }

} // namespace LIBC_NAMESPACE_DECL
