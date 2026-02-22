//===-- include/flang/Runtime/coarray.h --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_COARRAY_H
#define FORTRAN_RUNTIME_COARRAY_H

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
// class Descriptor;
extern "C" {

void RTDECL(ComputeLastUcobound)(
    int num_images, const Descriptor &lcobounds, const Descriptor &ucobounds);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COARRAY_H
