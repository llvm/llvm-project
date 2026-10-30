//===-- Unittests for rdivi -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FxDiviTest.h"

#include "llvm-libc-macros/stdfix-macros.h"
#include "src/stdfix/rdivi.h"

LIST_FXDIVI_TESTS(r, fract, int, LIBC_NAMESPACE::rdivi);
