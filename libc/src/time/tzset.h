//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the tzset function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TZSET_H
#define LLVM_LIBC_SRC_TIME_TZSET_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void tzset();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TZSET_H
