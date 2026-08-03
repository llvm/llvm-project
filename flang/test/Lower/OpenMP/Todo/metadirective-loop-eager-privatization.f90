! Loop-associated metadirective variants currently require delayed
! privatization.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - %s 2>&1 \
! RUN:   | FileCheck %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE with eager privatization

subroutine test_eager_privatization(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
