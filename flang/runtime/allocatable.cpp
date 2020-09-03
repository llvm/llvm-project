//===-- runtime/allocatable.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocatable.h"
#include "terminator.h"

namespace Fortran::runtime {
extern "C" {

void RTNAME(AllocatableInitIntrinsic)(
    Descriptor &, TypeCategory, int /*kind*/, int /*rank*/, int /*corank*/) {
  // TODO
}

void RTNAME(AllocatableInitCharacter)(Descriptor &, SubscriptValue /*length*/,
    int /*kind*/, int /*rank*/, int /*corank*/) {
  // TODO
}

void RTNAME(AllocatableInitDerived)(
    Descriptor &, const DerivedType &, int /*rank*/, int /*corank*/) {
  // TODO
}

void RTNAME(AllocatableAssign)(Descriptor &to, const Descriptor & /*from*/) {}

int RTNAME(MoveAlloc)(Descriptor &to, const Descriptor & /*from*/,
    bool /*hasStat*/, Descriptor * /*errMsg*/, const char * /*sourceFile*/,
    int /*sourceLine*/) {
  // TODO
  return 0;
}

int RTNAME(AllocatableDeallocate)(Descriptor &, bool /*hasStat*/,
    Descriptor * /*errMsg*/, const char * /*sourceFile*/, int /*sourceLine*/) {
  // TODO
  return 0;
}
}
} // namespace Fortran::runtime
