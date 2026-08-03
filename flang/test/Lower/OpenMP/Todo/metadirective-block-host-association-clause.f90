! Metadirective block variants do not have variant-local host-association
! symbols. Reject clauses that require them in both delayed and eager modes.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/firstprivate.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - \
! RUN:   %t/firstprivate.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/copyin.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - \
! RUN:   %t/copyin.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: METADIRECTIVE block variant with a clause requiring variant-local host association

!--- firstprivate.f90
subroutine firstprivate_block_variant(x)
  integer :: x
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel firstprivate(x)) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine

!--- copyin.f90
subroutine copyin_block_variant()
  integer, save :: x
  !$omp threadprivate(x)
  !$omp begin metadirective &
  !$omp& when(implementation={vendor(llvm)}: parallel copyin(x)) &
  !$omp& otherwise(nothing)
  x = x + 1
  !$omp end metadirective
end subroutine
