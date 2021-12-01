//===-- runtime/support.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/support.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {
extern "C" {

bool RTNAME(IsContiguous)(const Descriptor &descriptor) {
  return descriptor.IsContiguous();
}

} // extern "C"
} // namespace Fortran::runtime
