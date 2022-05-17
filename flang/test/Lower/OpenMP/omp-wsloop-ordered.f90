! This test checks lowering of worksharing-loop construct with ordered clause.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! This checks lowering ordered clause specified without parameter
subroutine wsloop_ordered_no_para()
  integer :: a(10), i

! CHECK:  omp.wsloop ordered(0) for (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:    omp.yield
! CHECK:  }

  !$omp do ordered
  do i = 2, 10
    !$omp ordered
    a(i) = a(i-1) + 1
    !$omp end ordered
  end do
  !$omp end do

end

! This checks lowering ordered clause specified with a parameter
subroutine wsloop_ordered_with_para()
  integer :: a(10), i

! CHECK: func @_QPwsloop_ordered_with_para() {
! CHECK:  omp.wsloop ordered(1) for (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:    omp.yield
! CHECK:  }

  !$omp do ordered(1)
  do i = 2, 10
    !!$omp ordered depend(sink: i-1)
    a(i) = a(i-1) + 1
    !!$omp ordered depend(source)
  end do
  !$omp end do

end
