//===-- include/flang/Runtime/AMD/amd_alloc.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_AMD_UMPIRE_H_
#define FORTRAN_RUNTIME_AMD_UMPIRE_H_

// TODO: check of the following two includes are necessary:
#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::amd {

extern "C" {
void RTDECL(AMDRegisterAllocator)();
void RTDECL(AMDAllocatableSetAllocIdx)(Descriptor &descriptor, int pos);
}

} // namespace Fortran::runtime::amd
#endif // FORTRAN_RUNTIME_AMD_UMPIRE_H_
