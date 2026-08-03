! Some loop iteration variables require construct-scoped name resolution for
! the private, linear, or lastprivate bindings of a selected loop variant.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/associate-do.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ASSOCIATE %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/associate-simd-collapse.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ASSOCIATE %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/allocatable.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=INDIRECT %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/pointer.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=INDIRECT %s

! ASSOCIATE: not yet implemented: ASSOCIATE name loop iteration variable in loop-associated METADIRECTIVE variant
! INDIRECT: not yet implemented: POINTER or ALLOCATABLE loop iteration variable in loop-associated METADIRECTIVE variant

!--- associate-do.f90
subroutine test_do(n, a)
  integer :: n, a(n), source_i
  associate(i => source_i)
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: do) &
    !$omp & otherwise(nothing)
    do i = 1, n
      a(i) = i
    end do
  end associate
end subroutine

!--- associate-simd-collapse.f90
subroutine test_simd_collapse(n, a)
  integer :: n, a(n, n), source_i, j
  associate(i => source_i)
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: simd collapse(2)) &
    !$omp & otherwise(nothing)
    do i = 1, n
      do j = 1, n
        a(j, i) = i + j
      end do
    end do
  end associate
end subroutine

!--- allocatable.f90
subroutine test_allocatable_iv(n, a)
  integer :: n, a(n)
  integer, allocatable :: i
  allocate(i)
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- pointer.f90
subroutine test_pointer_iv(n, a)
  integer :: n, a(n)
  integer, target :: target
  integer, pointer :: i
  i => target
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
