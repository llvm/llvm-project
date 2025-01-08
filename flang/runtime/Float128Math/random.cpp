//===-- runtime/Float128Math/random.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"
#include "numeric-template-specs.h"
#include "random-templates.h"

using namespace Fortran::runtime::random;
extern "C" {

#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(RandomNumber16)(
    const Descriptor &harvest, const char *source, int line) {
  return GenerateReal<CppTypeFor<TypeCategory::Real, 16>, 113>(harvest);
}
#endif

} // extern "C"
