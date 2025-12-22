//===-- lib/runtime/copy.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Utilities that copy data in a type-aware fashion, allocating & duplicating
// allocatable/automatic components of derived types along the way.

#ifndef FLANG_RT_RUNTIME_COPY_H_
#define FLANG_RT_RUNTIME_COPY_H_

#include "flang-rt/runtime/descriptor.h"

namespace Fortran::runtime {

// Assigns to uninitialized storage.
// Duplicates allocatable & automatic components.
RT_API_ATTRS void CopyElement(const Descriptor &to, const SubscriptValue toAt[],
    const Descriptor &from, const SubscriptValue fromAt[], Terminator &);

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_COPY_H_
