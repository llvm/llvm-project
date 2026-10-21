!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fsyntax-only %openmp_flags -fopenmp-version=52 %s
! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %s | FileCheck %s --check-prefix=HLFIR
! RUN: %flang_fc1 -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s | FileCheck %s --check-prefix=LLVM

subroutine scope_equivalence()
  integer :: x, y
  equivalence (x, y)
  !$omp scope private(x, y) allocate(x, y)
    x = 1
    call consume(x)
    y = 2
    call consume(y)
  !$omp end scope
end subroutine

! HLFIR-LABEL: func.func @_QPscope_equivalence
! HLFIR: omp.scope allocate(
! HLFIR-SAME: allocate_private_indices([0, 1])
! LLVM-LABEL: define void @scope_equivalence_
! LLVM: call ptr @__kmpc_alloc
! LLVM: call ptr @__kmpc_alloc

subroutine parallel_equivalence()
  integer :: x, y
  equivalence (x, y)
  !$omp parallel private(x, y) allocate(x, y)
    x = 1
    call consume(x)
    y = 2
    call consume(y)
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPparallel_equivalence
! HLFIR: omp.parallel allocate(
! HLFIR-SAME: allocate_private_indices([0, 1])
! LLVM-LABEL: define internal void @parallel_equivalence_..omp_par
! LLVM: call ptr @__kmpc_alloc
! LLVM: call ptr @__kmpc_alloc
