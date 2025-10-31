! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine do_concurrent_allocatable
  integer :: i
  real, allocatable, dimension(:,:) :: x

  do concurrent (i = 1:10) reduce(+: x)
  end do
end subroutine

! CHECK: fir.declare_reduction @[[RED_OP:.*]] : ![[RED_TYPE:.*]] alloc {
! CHECK:   %[[ALLOC:.*]] = fir.alloca
! CHECK:   fir.yield(%[[ALLOC]] : ![[RED_TYPE]])
! CHECK: } init {
! CHECK: ^bb0(%{{.*}}: ![[RED_TYPE]], %[[RED_ARG:.*]]: ![[RED_TYPE]]):
! CHECK:   fir.yield(%[[RED_ARG]] : !{{.*}})
! CHECK: } combiner {
! CHECK: ^bb0(%[[COMB_RES:.*]]: ![[RED_TYPE]], %{{.*}}: ![[RED_TYPE]]):
! CHECK:   fir.yield(%[[COMB_RES]] : !{{.*}})
! CHECK: } cleanup {
! CHECK:   fir.yield
! CHECK: }
