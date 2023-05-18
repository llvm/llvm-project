//===-- include/flang/Runtime/temporary-stack.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Runtime functions for storing a dynamically resizable number of temporaries.
// For use in HLFIR lowering.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TEMPORARY_STACK_H_
#define FORTRAN_RUNTIME_TEMPORARY_STACK_H_

#include "flang/Runtime/entry-names.h"
#include <stdint.h>

namespace Fortran::runtime {
class Descriptor;
extern "C" {

// Stores both the descriptor and a copy of the value in a dynamically resizable
// data structure identified by opaquePtr. All value stacks must be destroyed
// at the end of their lifetime and not used afterwards.
// Popped descriptors point to the copy of the value, not the original address
// of the value. This copy is dynamically allocated, it is up to the caller to
// free the value pointed to by the box. The copy operation is a simple memcpy.
// The sourceFile and line number used when creating the stack are shared for
// all operations.
// Opaque pointers returned from these are incompatible with those returned by
// the flavours for storing descriptors.
[[nodiscard]] void *RTNAME(CreateValueStack)(
    const char *sourceFile = nullptr, int line = 0);
void RTNAME(PushValue)(void *opaquePtr, const Descriptor &value);
// Note: retValue should be large enough to hold the right number of dimensions,
// and the optional descriptor addendum
void RTNAME(PopValue)(void *opaquePtr, Descriptor &retValue);
// Return the i'th element into retValue (which must be the right size). An
// exact copy of this descriptor remains in this storage so this one should not
// be deallocated
void RTNAME(ValueAt)(void *opaquePtr, uint64_t i, Descriptor &retValue);
void RTNAME(DestroyValueStack)(void *opaquePtr);

// Stores descriptors value in a dynamically resizable data structure identified
// by opaquePtr. All descriptor stacks must be destroyed at the end of their
// lifetime and not used afterwards.
// Popped descriptors are identical to those which were pushed.
// The sourceFile and line number used when creating the stack are shared for
// all operations.
// Opaque pointers returned from these are incompatible with those returned by
// the flavours for storing both descriptors and values.
[[nodiscard]] void *RTNAME(CreateDescriptorStack)(
    const char *sourceFile = nullptr, int line = 0);
void RTNAME(PushDescriptor)(void *opaquePtr, const Descriptor &value);
// Note: retValue should be large enough to hold the right number of dimensions,
// and the optional descriptor addendum
void RTNAME(PopDescriptor)(void *opaquePtr, Descriptor &retValue);
// Return the i'th element into retValue (which must be the right size). An
// exact copy of this descriptor remains in this storage so this one should not
// be deallocated
void RTNAME(DescriptorAt)(void *opaquePtr, uint64_t i, Descriptor &retValue);
void RTNAME(DestroyDescriptorStack)(void *opaquePtr);

} // extern "C"
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_TEMPORARY_STACK_H_
