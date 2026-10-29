//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the application properties
///
//===----------------------------------------------------------------------===//

#include "config/app.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef LIBC_COPT_SHARED_LIBRARY
// The startup code fills this in, but libc.so reads it, and an executable
// does not export its symbols to the objects it loads. So in a shared build
// the definition lives here, inside libc.so, where both can reach it. The
// static build keeps it in crt1 with the code that populates it.
AppProperties app;
#endif

} // namespace LIBC_NAMESPACE_DECL
