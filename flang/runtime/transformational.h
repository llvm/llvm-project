//===-- runtime/transformational.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API for the type-independent transformational intrinsic functions
// that rearrange data in arrays: CSHIFT, EOSHIFT, PACK, RESHAPE, SPREAD,
// TRANSPOSE, and UNPACK.
// These are naive allocating implementations; optimized forms that manipulate
// pointer descriptors or that supply functional views of arrays remain to
// be defined and may instead be part of lowering (see docs/ArrayComposition.md)
// for details).

#ifndef FORTRAN_RUNTIME_TRANSFORMATIONAL_H_
#define FORTRAN_RUNTIME_TRANSFORMATIONAL_H_

#include "descriptor.h"
#include "entry-names.h"
#include "memory.h"

namespace Fortran::runtime {

extern "C" {

void RTNAME(Reshape)(Descriptor &result, const Descriptor &source,
    const Descriptor &shape, const Descriptor *pad = nullptr,
    const Descriptor *order = nullptr, const char *sourceFile = nullptr,
    int line = 0);

void RTNAME(Cshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, int dim = 1, const char *sourceFile = nullptr,
    int line = 0);
void RTNAME(CshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const char *sourceFile = nullptr, int line = 0);

void RTNAME(Eoshift)(Descriptor &result, const Descriptor &source,
    const Descriptor &shift, const Descriptor *boundary = nullptr, int dim = 1,
    const char *sourceFile = nullptr, int line = 0);
void RTNAME(EoshiftVector)(Descriptor &result, const Descriptor &source,
    std::int64_t shift, const Descriptor *boundary = nullptr,
    const char *sourceFile = nullptr, int line = 0);

void RTNAME(Pack)(Descriptor &result, const Descriptor &source,
    const Descriptor &mask, const Descriptor *vector = nullptr,
    const char *sourceFile = nullptr, int line = 0);

void RTNAME(Spread)(Descriptor &result, const Descriptor &source, int dim,
    std::int64_t ncopies, const char *sourceFile = nullptr, int line = 0);

void RTNAME(Transpose)(Descriptor &result, const Descriptor &matrix,
    const char *sourceFile = nullptr, int line = 0);

void RTNAME(Unpack)(Descriptor &result, const Descriptor &vector,
    const Descriptor &mask, const Descriptor &field,
    const char *sourceFile = nullptr, int line = 0);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TRANSFORMATIONAL_H_
