! Verify that the omp_lib.h header is found and included correctly. This header file should be available at a path:
!   * relative to the driver, that's
!   * known the driver.
! This is taken care of at the CMake and the driver levels. Note that when searching for header files, the directory of the current
! source file takes precedence over other search paths. Hence adding omp_lib.h in the current directory will make Flang use that
! header file instead of the one shipped with Flang.

! This should just work
! RUN: not rm omp_lib.h
! RUN: %flang -fsyntax-only -fopenmp %s  2>&1

! Create an empty omp_lib.h header that _does not_ define omp_default_mem_alloc - this should lead to semantic errors
! RUN: touch omp_lib.h
! RUN: not %flang -fsyntax-only -fopenmp %s  2>&1 | FileCheck %s
! RUN: rm omp_lib.h

! CHECK: error: Must have INTEGER type, but is REAL(4)

include "omp_lib.h"

integer :: x, y

!$omp allocate(x, y) allocator(omp_default_mem_alloc)

end
