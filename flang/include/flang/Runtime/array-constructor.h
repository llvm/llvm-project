//===-- include/flang/Runtime/array-constructor.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External APIs to create temporary storage for array constructors when their
// final extents or length parameters cannot be pre-computed.

#ifndef FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_
#define FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_

#include "flang/Runtime/array-constructor-consts.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {

// Runtime data structure to hold information about the storage of
// an array constructor being constructed.
struct ArrayConstructorVector {
  RT_API_ATTRS ArrayConstructorVector(class Descriptor &to,
      SubscriptValue nextValuePosition, SubscriptValue actualAllocationSize,
      const char *sourceFile, int sourceLine, bool useValueLengthParameters)
      : to{to}, nextValuePosition{nextValuePosition},
        actualAllocationSize{actualAllocationSize}, sourceFile{sourceFile},
        sourceLine{sourceLine},
        useValueLengthParameters_{useValueLengthParameters} {}

  RT_API_ATTRS bool useValueLengthParameters() const {
    return useValueLengthParameters_;
  }

  class Descriptor &to;
  SubscriptValue nextValuePosition;
  SubscriptValue actualAllocationSize;
  const char *sourceFile;
  int sourceLine;

private:
  unsigned char useValueLengthParameters_ : 1;
};

static_assert(sizeof(Fortran::runtime::ArrayConstructorVector) <=
        MaxArrayConstructorVectorSizeInBytes,
    "ABI requires sizeof(ArrayConstructorVector) to be smaller than "
    "MaxArrayConstructorVectorSizeInBytes");
static_assert(alignof(Fortran::runtime::ArrayConstructorVector) <=
        MaxArrayConstructorVectorAlignInBytes,
    "ABI requires alignof(ArrayConstructorVector) to be smaller than "
    "MaxArrayConstructorVectorAlignInBytes");

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ARRAYCONSTRUCTOR_H_
