//===-- Implementation of dlopen -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlopen.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO(@izaakschroeder): https://github.com/llvm/llvm-project/issues/97919
LLVM_LIBC_FUNCTION(void *, dlopen, (const char *, int)) { return nullptr; }

} // namespace LIBC_NAMESPACE_DECL
