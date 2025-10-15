! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s

! Check that we can lower this without crashing

! CHECK: func.func @_QPlastprivate_of_private
subroutine lastprivate_of_private(a)
  real :: a(100)
  integer i
  ! CHECK: omp.parallel private({{.*}}) {
  !$omp parallel private(a)
    ! CHECK: omp.parallel {
    !$omp parallel shared(a)
    ! CHECK: omp.wsloop private({{.*}}) {
    !$omp do lastprivate(a)
    ! CHECK: omp.loop_nest
      do i=1,100
        a(i) = 1.0
      end do
    !$omp end parallel
  !$omp end parallel
end subroutine
