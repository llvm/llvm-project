//===-- Unittests for lrdivi ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FxDiviTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/lrdivi.h"

LIST_FXDIVI_TESTS(lr, long fract, long int, LIBC_NAMESPACE::lrdivi);
