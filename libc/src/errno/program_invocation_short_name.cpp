//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of program_invocation_short_name.
///
//===----------------------------------------------------------------------===//

#include "src/errno/program_invocation_short_name.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Initialized by startup code.
LLVM_LIBC_VARIABLE(char *, program_invocation_short_name) = nullptr;

} // namespace LIBC_NAMESPACE_DECL
