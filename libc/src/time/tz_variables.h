//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the tzname, timezone, and daylight
/// variables.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TZNAME_H
#define LLVM_LIBC_SRC_TIME_TZNAME_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern char *tzname[2];
extern long timezone;
extern int daylight;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TZNAME_H
