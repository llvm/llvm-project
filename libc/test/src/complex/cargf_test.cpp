//===-- Unittests for cargf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CargTest.h"

#include "src/complex/cargf.h"

LIST_CARG_TESTS(_Complex float, float, LIBC_NAMESPACE::cargf)
