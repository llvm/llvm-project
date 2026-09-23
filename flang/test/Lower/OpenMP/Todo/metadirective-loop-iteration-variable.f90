! Some loop iteration variables require construct-scoped name resolution for
! the private, linear, or lastprivate bindings of a selected loop variant.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/associate-do.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ASSOCIATE %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/pointer.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=INDIRECT %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/threadprivate.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=THREADPRIVATE %s

! ASSOCIATE: not yet implemented: ASSOCIATE name loop iteration variable in loop-associated METADIRECTIVE variant
! INDIRECT: not yet implemented: POINTER or ALLOCATABLE loop iteration variable in loop-associated METADIRECTIVE variant
! THREADPRIVATE: not yet implemented: THREADPRIVATE loop iteration variable in loop-associated METADIRECTIVE variant

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

!--- threadprivate.f90
subroutine test_threadprivate_iv(n, a)
  integer :: n, a(n)
  integer, save :: i
  !$omp threadprivate(i)
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
