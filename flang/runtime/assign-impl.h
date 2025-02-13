//===-- runtime/assign-impl.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ASSIGN_IMPL_H_
#define FORTRAN_RUNTIME_ASSIGN_IMPL_H_

#include "flang/Runtime/freestanding-tools.h"

namespace Fortran::runtime {
class Descriptor;
class Terminator;

// Assign one object to another via allocate statement from source specifier.
// Note that if allocate object and source expression have the same rank, the
// value of the allocate object becomes the value provided; otherwise the value
// of each element of allocate object becomes the value provided (9.7.1.2(7)).
#ifdef RT_DEVICE_COMPILATION
RT_API_ATTRS void DoFromSourceAssign(Descriptor &, const Descriptor &,
    Terminator &, MemmoveFct memmoveFct = &MemmoveWrapper);
#else
RT_API_ATTRS void DoFromSourceAssign(Descriptor &, const Descriptor &,
    Terminator &, MemmoveFct memmoveFct = &Fortran::runtime::memmove);
#endif

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ASSIGN_IMPL_H_
