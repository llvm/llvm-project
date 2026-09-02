//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of _r_debug.
///
//===----------------------------------------------------------------------===//

#include "src/link/_r_debug.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Initialized by startup code.
LLVM_LIBC_VARIABLE(struct r_debug, _r_debug) = {};

} // namespace LIBC_NAMESPACE_DECL
