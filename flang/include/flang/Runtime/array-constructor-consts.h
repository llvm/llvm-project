//===-- include/flang/Runtime/array-constructor-consts.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External APIs to create temporary storage for array constructors when their
// final extents or length parameters cannot be pre-computed.

#ifndef FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_
#define FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {
struct ArrayConstructorVector;

// Max sizeof(ArrayConstructorVector) and sizeof(ArrayConstructorVector) for any
// target.
// TODO: Use target-specific size/alignment instead of overapproximation.
constexpr std::size_t MaxArrayConstructorVectorSizeInBytes = 2 * 40;
constexpr std::size_t MaxArrayConstructorVectorAlignInBytes = 8;

extern "C" {
// API to initialize an ArrayConstructorVector before any values are pushed to
// it. Inlined code is only expected to allocate the "ArrayConstructorVector"
// class instance storage with sufficient size (using
// "2*sizeof(ArrayConstructorVector)" on the host should be safe regardless of
// the target the runtime is compiled for). This avoids the need for the runtime
// to maintain a state, or to use dynamic allocation for it. "vectorClassSize"
// is used to validate that lowering allocated enough space for it.
void RTDECL(InitArrayConstructorVector)(ArrayConstructorVector &vector,
    Descriptor &to, bool useValueLengthParameters, int vectorClassSize,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Generic API to push any kind of entity into the array constructor (any
// Fortran type and any rank).
void RTDECL(PushArrayConstructorValue)(
    ArrayConstructorVector &vector, const Descriptor &from);

// API to push scalar array constructor value of:
//   - a numerical or logical type,
//   - or a derived type that has no length parameters, and no allocatable
//   component (that would require deep copies).
// It requires no descriptor for the value that is passed via its base address.
void RTDECL(PushArrayConstructorSimpleScalar)(
    ArrayConstructorVector &vector, void *from);
} // extern "C"
} // namespace Fortran::runtime
#endif /* FORTRAN_RUNTIME_ARRAY_CONSTRUCTOR_CONSTS_H_ */
