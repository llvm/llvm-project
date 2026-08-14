//===-- include/flang/Runtime/OpenMP/omp_alloc.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_OMP_ALLOC_H_
#define FORTRAN_RUNTIME_OMP_ALLOC_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::omp {

extern "C" {

/// Register the OpenMP target device allocator with the Fortran runtime's
/// allocator registry.  Called once from the generated main() when
/// -fopenmp-default-allocate=target is active.  The allocator uses
/// omp_target_alloc/omp_target_free to place Fortran ALLOCATABLE storage
/// on the current default device.  The environment variable OMP_ALLOC
/// (default: "openmp") selects the allocator backend; OMP_ALLOC_DEBUG
/// enables diagnostic tracing to stderr.
void RTDECL(OpenMPRegisterAllocator)();

/// Set the allocator index on an allocatable descriptor so that subsequent
/// AllocatableAllocate calls route through the registered OpenMP allocator.
/// \p descriptor must be an unallocated ALLOCATABLE; \p pos is the allocator
/// registry slot (typically 1).  No-op if the descriptor is already allocated
/// or is not allocatable.
void RTDECL(OpenMPAllocatableSetAllocIdx)(Descriptor &descriptor, int pos);

}

} // namespace Fortran::runtime::omp
#endif // FORTRAN_RUNTIME_OMP_ALLOC_H_
