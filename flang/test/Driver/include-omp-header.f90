! REQUIRES: openmp_runtime

! Check omp_lib.h works with driver
! RUN: %flang -fsyntax-only -cpp %s -v 2>&1 | FileCheck %s --check-prefix=DRIVER
! RUN: %flang -fsyntax-only -cpp %s -v -DHASHINCLUDE 2>&1 | FileCheck %s --check-prefix=DRIVER
! DRIVER: -fc1
! DRIVER-SAME: -fintrinsic-modules-path

! Check frontend only works (no output expected)
! RUN: %flang_fc1 -fsyntax-only -cpp %s
! RUN: %flang_fc1 -fsyntax-only -cpp -DHASHINCLUDE %s

! Check omp_lib.h contents
! RUN: %flang_fc1 -cpp %s -E -fno-reformat 2>&1 | FileCheck %s --check-prefix=PREPROCESSED
! RUN: %flang_fc1 -cpp %s -E -fno-reformat -DHASHINCLUDE 2>&1 | FileCheck %s --check-prefix=PREPROCESSED
! PREPROCESSED: integer(kind=omp_integer_kind)openmp_version
! PREPROCESSED: parameter(openmp_version={{[0-9]+}})


program main
#ifdef HASHINCLUDE
  #include "omp_lib.h"
#else
  include "omp_lib.h"
#endif

  integer :: x, y
  !$omp allocate(x, y) allocator(omp_default_mem_alloc)

  print *, 'PASS: openmp_version parameter ', openmp_version
end program main
