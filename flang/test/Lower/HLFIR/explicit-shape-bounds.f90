! Test lowering of explicit-shape bounds using rank-1 integer arrays
! (RankOneBoundElement in the evaluate representation).
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! Test with PARAMETER rank-1 bounds: the constant array is folded so the
! explicit-shape bounds become compile-time constants (no runtime load).
module test_param
contains
  subroutine test_param_bounds()
    integer, parameter :: dims(3) = [2, 3, 4]
    real :: a(dims)
    a(1,1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_paramPtest_param_bounds()
! CHECK:  %[[C2:.*]] = arith.constant 2 : index
! CHECK:  %[[C3:.*]] = arith.constant 3 : index
! CHECK:  %[[C4:.*]] = arith.constant 4 : index
! CHECK:  fir.alloca !fir.array<2x3x4xf32>
! CHECK:  fir.shape %[[C2]], %[[C3]], %[[C4]] : (index, index, index) -> !fir.shape<3>

! Test with rank-1 dummy as upper bounds only.
module test_dummy_upper
contains
  subroutine test_dummy_upper_bounds(n)
    integer, intent(in) :: n(3)
    real :: a(n)
    a(1,1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_dummy_upperPtest_dummy_upper_bounds(
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<3xi64>
! CHECK:  ^bb0(%arg{{.*}}: index):
! CHECK:    hlfir.designate {{.*}} (%arg{{.*}}) : ({{.*}}, index) -> !fir.ref<i32>
! CHECK:    fir.load {{.*}} : !fir.ref<i32>
! CHECK:    fir.convert {{.*}} : (i32) -> i64
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply {{.*}}, %[[C1]] : (!hlfir.expr<3xi64>, index) -> i64
! CHECK:  %[[C2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply {{.*}}, %[[C2]] : (!hlfir.expr<3xi64>, index) -> i64
! CHECK:  %[[C3:.*]] = arith.constant 3 : index
! CHECK:  hlfir.apply {{.*}}, %[[C3]] : (!hlfir.expr<3xi64>, index) -> i64

! Test with both lower and upper rank-1 bounds.
module test_dummy_both
contains
  subroutine test_dummy_both_bounds(lb, ub)
    integer, intent(in) :: lb(2), ub(2)
    real :: a(lb:ub)
    a(1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_dummy_bothPtest_dummy_both_bounds(
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply {{.*}}, %[[C1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[C1_1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply {{.*}}, %[[C1_1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[C2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply {{.*}}, %[[C2]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[C2_1:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply {{.*}}, %[[C2_1]] : (!hlfir.expr<2xi64>, index) -> i64

! Test broadcast of scalar lower bound with rank-1 upper bounds.
module test_broadcast
contains
  subroutine test_broadcast_bounds(ub)
    integer, intent(in) :: ub(2)
    real :: a(0:ub)
    a(0,0) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_broadcastPtest_broadcast_bounds(
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[U1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply {{.*}}, %[[U1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[U2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply {{.*}}, %[[U2]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  fir.shape_shift {{.*}} : (index, index, index, index) -> !fir.shapeshift<2>

! Test rank-1 dummy bounds of explicit integer kinds: the bound base is always
! coerced to a 64-bit subscript, so INTEGER(4) is widened (element loaded as i32
! then converted to i64) while INTEGER(8) is already 64-bit and loaded directly.
module test_bound_kinds
contains
  subroutine test_bound_kinds_bounds(n4, n8)
    integer(4), intent(in) :: n4(2)
    integer(8), intent(in) :: n8(2)
    real :: a4(n4)
    real :: a8(n8)
    a4(1,1) = 1.0
    a8(1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_bound_kindsPtest_bound_kinds_bounds(
! CHECK:  hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:    hlfir.designate {{.*}} (%arg{{.*}}) : ({{.*}}, index) -> !fir.ref<i32>
! CHECK:    fir.load {{.*}} : !fir.ref<i32>
! CHECK:    fir.convert {{.*}} : (i32) -> i64
! CHECK:  hlfir.designate {{.*}} : ({{.*}}, index) -> !fir.ref<i64>
! CHECK:  fir.load {{.*}} : !fir.ref<i64>
