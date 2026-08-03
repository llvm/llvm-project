! ASSOCIATE-name loop iteration variables require construct-scoped name
! resolution for private and lastprivate bindings.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/do.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/simd-collapse.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: ASSOCIATE name loop iteration variable in loop-associated METADIRECTIVE variant

!--- do.f90
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

!--- simd-collapse.f90
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
