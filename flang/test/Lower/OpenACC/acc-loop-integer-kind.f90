! DO loop parameters are converted to the induction variable's kind
! (F2023 11.1.7.4.1). An acc.loop carries those converted values, including
! an omitted step of 1 in the induction variable's kind.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! i32 induction variable, integer(8) parameter bound, omitted step, tile.
subroutine tiled_i64_param()
  integer(8), parameter :: n = 100
  integer :: i
  real :: a(n)
  !$acc parallel loop tile(8)
  do i = 1, n
    a(i) = 1.0
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtiled_i64_param
! CHECK: %[[UB:.*]] = fir.convert {{.*}} : (i64) -> i32
! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
! CHECK: acc.loop {{.*}} tile({%{{.*}} : i32}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%[[UB]] : i32) step (%[[STEP]] : i32)

! i32 induction variable, integer(8) variable bound and step.
subroutine i64_bound_and_step(n, s)
  integer(8) :: n, s
  integer :: i
  real :: a(10)
  !$acc parallel loop
  do i = 1, n, s
    a(i) = 1.0
  end do
end subroutine

! CHECK-LABEL: func.func @_QPi64_bound_and_step
! CHECK: %[[LB:.*]] = arith.constant 1 : i32
! CHECK: %[[N:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK: %[[UB:.*]] = fir.convert %[[N]] : (i64) -> i32
! CHECK: %[[S:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK: %[[STEP:.*]] = fir.convert %[[S]] : (i64) -> i32
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%[[LB]] : i32) to (%[[UB]] : i32) step (%[[STEP]] : i32)

! integer(8) induction variable, default-kind bounds, omitted step.
subroutine i64_iv_default_bounds()
  integer(8) :: i
  real :: a(10)
  !$acc kernels
  do i = 1, 10
    a(i) = 1.0
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPi64_iv_default_bounds
! CHECK: %[[LB:.*]] = fir.convert {{.*}} : (i32) -> i64
! CHECK: %[[UB:.*]] = fir.convert {{.*}} : (i32) -> i64
! CHECK: %[[STEP:.*]] = arith.constant 1 : i64
! CHECK: acc.loop {{.*}} control(%{{.*}} : i64) = (%[[LB]] : i64) to (%[[UB]] : i64) step (%[[STEP]] : i64)

! DO CONCURRENT index is default integer; the limit is integer(8).
subroutine concurrent_i64_limit(n)
  integer(8) :: n
  integer :: i
  real :: a(10)
  !$acc kernels
  do concurrent (i = 1:n)
    a(i) = 1.0
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPconcurrent_i64_limit
! CHECK: %[[UB:.*]] = fir.convert {{.*}} : (i64) -> i32
! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%[[UB]] : i32) step (%[[STEP]] : i32)

! Matching kinds stay unconverted: the control values are already i64.
subroutine matching_i64()
  integer(8) :: i
  integer(8), parameter :: n = 10
  real :: a(n)
  !$acc kernels
  do i = 1_8, n, 1_8
    a(i) = 1.0
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPmatching_i64
! CHECK-NOT: fir.convert {{.*}} : (i64) -> i64
! CHECK: acc.loop {{.*}} control(%{{.*}} : i64) = (%{{.*}} : i64) to (%{{.*}} : i64) step (%{{.*}} : i64)
