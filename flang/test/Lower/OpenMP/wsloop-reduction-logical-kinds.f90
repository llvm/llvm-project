! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! Check that logical reductions of different kinds do not end up using the same
! reduction declaration

! CHECK-LABEL:   omp.declare_reduction @or_reduction_l64 : !fir.logical<8> init {
! CHECK-LABEL:   omp.declare_reduction @or_reduction_l32 : !fir.logical<4> init {

subroutine test(a4, a8, sz)
  integer :: sz
  logical(4), dimension(sz) :: a4
  logical(8), dimension(sz) :: a8

  logical(4) :: res4 = .false.
  logical(8) :: res8 = .false.
  integer i

! CHECK: omp.wsloop private({{.*}}) reduction(@or_reduction_l32 {{.*}}, @or_reduction_l64 {{.*}}) {
  !$omp do reduction(.or.:res4, res8)
  do i = 1,sz
    res4 = res4 .or. a4(i)
    res8 = res8 .or. a8(i)
  enddo
end subroutine
