//===-- runtime/transformational.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TRANSFORMATIONAL_H_
#define FORTRAN_RUNTIME_TRANSFORMATIONAL_H_

#include "descriptor.h"
#include "memory.h"

namespace Fortran::runtime {

OwningPtr<Descriptor> RESHAPE(const Descriptor &source, const Descriptor &shape,
    const Descriptor *pad = nullptr, const Descriptor *order = nullptr);
}
#endif // FORTRAN_RUNTIME_TRANSFORMATIONAL_H_
