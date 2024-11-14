! Tests that there no debug-info conflicts arise because of DI attached to nested
! OMP regions arguments.

! RUN: %flang -c -fopenmp -g -mmlir --openmp-enable-delayed-privatization=true \
! RUN:   %s -o - 2>&1 | FileCheck %s

subroutine bar (b)
  integer :: a, b
!$omp parallel
  do a = 1, 10
    b = a
  end do
!$omp end parallel
end subroutine bar

! CHECK-NOT: conflicting debug info for argument
