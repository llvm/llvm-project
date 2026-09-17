! Test lowering of explicit-shape bounds using rank-1 integer arrays
! (RankOneBoundElement in the evaluate representation).  A constant base folds
! to compile-time constants; otherwise the N bounds that share one rank-1 base
! are lowered by evaluating that base exactly once and extracting each
! dimension's element from it, rather than re-evaluating the base per dimension.
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! Test with PARAMETER rank-1 bounds: the shared constant array is folded, so
! each dimension's bound is a compile-time constant and no base array is
! materialized.
module test_param
contains
  subroutine test_param_bounds()
    integer, parameter :: dims(3) = [2, 3, 4]
    real :: a(dims)
    a(1,1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_paramPtest_param_bounds()
! CHECK-NOT: hlfir.designate
! CHECK:  %[[C2:.*]] = arith.constant 2 : index
! CHECK:  %[[C3:.*]] = arith.constant 3 : index
! CHECK:  %[[C4:.*]] = arith.constant 4 : index
! CHECK:  fir.alloca !fir.array<2x3x4xf32>
! CHECK:  fir.shape %[[C2]], %[[C3]], %[[C4]]

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
! CHECK:  %[[N:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<3xi64>
! CHECK:  ^bb0(%arg{{.*}}: index):
! CHECK:    hlfir.designate {{.*}} (%arg{{.*}}) : ({{.*}}, index) -> !fir.ref<i32>
! CHECK:    fir.load {{.*}} : !fir.ref<i32>
! CHECK:    fir.convert {{.*}} : (i32) -> i64
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply %[[N]], %[[C1]] : (!hlfir.expr<3xi64>, index) -> i64
! CHECK:  %[[C2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply %[[N]], %[[C2]] : (!hlfir.expr<3xi64>, index) -> i64
! CHECK:  %[[C3:.*]] = arith.constant 3 : index
! CHECK:  hlfir.apply %[[N]], %[[C3]] : (!hlfir.expr<3xi64>, index) -> i64

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
! CHECK:  %[[LB:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[L1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply %[[LB]], %[[L1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[L2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply %[[LB]], %[[L2]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[UB:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[U1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply %[[UB]], %[[U1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[U2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply %[[UB]], %[[U2]] : (!hlfir.expr<2xi64>, index) -> i64

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
! CHECK:  %[[UB:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[U1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply %[[UB]], %[[U1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[U2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply %[[UB]], %[[U2]] : (!hlfir.expr<2xi64>, index) -> i64
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
! CHECK:  %[[N8:.*]]:2 = hlfir.declare %arg1{{.*}}n8"} : (!fir.ref<!fir.array<2xi64>>{{.*}}
! CHECK:  %[[A4:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:    hlfir.designate {{.*}} (%arg{{.*}}) : ({{.*}}, index) -> !fir.ref<i32>
! CHECK:    fir.load {{.*}} : !fir.ref<i32>
! CHECK:    fir.convert {{.*}} : (i32) -> i64
! CHECK:  hlfir.apply %[[A4]], %{{.*}} : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  hlfir.apply %[[A4]], %{{.*}} : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  hlfir.designate %[[N8]]#0 (%{{.*}}){{.*}}-> !fir.ref<i64>
! CHECK:  fir.load {{.*}} : !fir.ref<i64>
! CHECK:  hlfir.designate %[[N8]]#0 (%{{.*}}){{.*}}-> !fir.ref<i64>
! CHECK:  fir.load {{.*}} : !fir.ref<i64>

! Test a polymorphic explicit-shape dummy: it is tracked through a descriptor,
! so the rank-1 upper bounds are lowered on the descriptor path (via
! lowerExplicitExtents) rather than the plain-array path.  The shared base is
! still evaluated once.
module test_poly_upper
contains
  subroutine test_poly_upper_bounds(n, a)
    integer, intent(in) :: n(3)
    class(*), intent(in) :: a(n)
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_poly_upperPtest_poly_upper_bounds(
! CHECK:  %[[N:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<3xi64>
! CHECK:    hlfir.designate {{.*}} (%arg{{.*}}) : ({{.*}}, index) -> !fir.ref<i32>
! CHECK:    fir.load {{.*}} : !fir.ref<i32>
! CHECK:    fir.convert {{.*}} : (i32) -> i64
! CHECK-NOT: hlfir.elemental
! CHECK:  hlfir.declare %arg1 {{.*}}!fir.class
! CHECK:  hlfir.destroy %[[N]]

! Test a polymorphic explicit-shape dummy with rank-1 lower and upper bounds:
! the descriptor path uses lowerExplicitLowerBounds and lowerExplicitExtents.
! Each side's shared base is evaluated once; the lower bounds feed fir.shift.
module test_poly_both
contains
  subroutine test_poly_both_bounds(lb, ub, a)
    integer, intent(in) :: lb(2), ub(2)
    class(*), intent(in) :: a(lb:ub)
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_poly_bothPtest_poly_both_bounds(
! CHECK:  %[[LB:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[L1:.*]] = arith.constant 1 : index
! CHECK:  hlfir.apply %[[LB]], %[[L1]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[L2:.*]] = arith.constant 2 : index
! CHECK:  hlfir.apply %[[LB]], %[[L2]] : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  %[[UB:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  fir.shift {{.*}} : (index, index) -> !fir.shift<2>
! CHECK:  hlfir.declare %arg2{{.*}}!fir.class

! Test a bound that embeds a rank-1 bound element inside other operations: the
! upper bound of `b` is ubound(a,2), and `a`'s base merge(n,m,c) is irreducible
! (logical mask), so the front end keeps the bound element wrapped in
! int(int(max(0, ROBE), 4), 8).  The shared-bound helper only matches a bound
! that is a bare rank-1 bound element, so this one falls back to generic
! expression lowering, which re-evaluates the base and extracts the element.
module test_ubound_robe
contains
  subroutine test_ubound_robe_bounds(n, m, c)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = 1.0
  end subroutine
end module
! CHECK-LABEL: func.func @_QMtest_ubound_robePtest_ubound_robe_bounds(
! CHECK:  hlfir.declare {{.*}}Ea"
! CHECK:  %[[UBBASE:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<2xi64>
! CHECK:  %[[UBELT:.*]] = hlfir.apply %[[UBBASE]], %{{.*}} : (!hlfir.expr<2xi64>, index) -> i64
! CHECK:  arith.maxsi %{{.*}}, %[[UBELT]]
! CHECK:  hlfir.declare {{.*}}Eb"
