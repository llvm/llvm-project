! Check that the DO variable of a loop lowered to an acc.loop without a loop
! directive holds its final value after the loop when that value may be used.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

function used_after_in_routine(n) result(last)
  !$acc routine seq
  integer :: n, k, last
  do k = 1, n
  end do
  last = k
end function

! CHECK-LABEL: func.func @_QPused_after_in_routine(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_in_routineEk")
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) = (%[[LB:.*]] : i32) to (%[[UB:.*]] : i32) step (%[[STEP:.*]] : i32) {
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LB_IDX:.*]] = fir.convert %[[LB]] : (i32) -> index
! CHECK: %[[UB_IDX:.*]] = fir.convert %[[UB]] : (i32) -> index
! CHECK: %[[STEP_IDX:.*]] = fir.convert %[[STEP]] : (i32) -> index
! CHECK: %[[ZERO:.*]] = arith.constant 0 : index
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB_IDX]], %[[LB_IDX]] : index
! CHECK: %[[SPAN:.*]] = arith.addi %[[DIFF]], %[[STEP_IDX]] : index
! CHECK: %[[QUOT:.*]] = arith.divsi %[[SPAN]], %[[STEP_IDX]] : index
! CHECK: %[[POSITIVE:.*]] = arith.cmpi sgt, %[[QUOT]], %[[ZERO]] : index
! CHECK: %[[COUNT:.*]] = arith.select %[[POSITIVE]], %[[QUOT]], %[[ZERO]] : index
! CHECK: %[[STRIDE:.*]] = arith.muli %[[COUNT]], %[[STEP_IDX]] : index
! CHECK: %[[LAST:.*]] = arith.addi %[[LB_IDX]], %[[STRIDE]] : index
! CHECK: %[[LAST_I32:.*]] = fir.convert %[[LAST]] : (index) -> i32
! CHECK: fir.store %[[LAST_I32]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>

! The bounds do not have the kind of the variable.
function mixed_kinds(n) result(last)
  !$acc routine seq
  integer :: n
  integer(8) :: k, last
  do k = 1, n
  end do
  last = k
end function

! CHECK-LABEL: func.func @_QPmixed_kinds(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFmixed_kindsEk")
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i64>) control(%{{.*}} : i64) =
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LB_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[UB_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[STEP_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB_IDX]], %[[LB_IDX]] : index
! CHECK: %[[STRIDE:.*]] = arith.muli %{{.*}}, %[[STEP_IDX]] : index
! CHECK: %[[LAST:.*]] = arith.addi %[[LB_IDX]], %[[STRIDE]] : index
! CHECK: %[[LAST_I64:.*]] = fir.convert %[[LAST]] : (index) -> i64
! CHECK: fir.store %[[LAST_I64]] to %[[K]]#0 : !fir.ref<i64>

subroutine used_after_in_serial(n, r)
  integer :: n, k, r
  !$acc serial copyout(r)
  do k = 1, n
  end do
  r = k
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPused_after_in_serial(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_in_serialEk")
! CHECK: acc.serial
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>

! The assignment is in the body of the enclosing loop, after the inner loop.
subroutine used_after_inner_loop(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel loop
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_after_inner_loop(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_inner_loopEk")
! CHECK: acc.loop combined(parallel) private(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The variable is private to the enclosing loop: its copy is assigned.
subroutine used_in_private_loop(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel loop private(k)
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_in_private_loop(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) name("k")
! CHECK: acc.loop combined(parallel) private(%[[K_PRIVATE]],
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFused_in_private_loopEk")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>

! The bounds of a later loop on the variable use its value.
subroutine used_by_next_loop(n, m, s)
  !$acc routine seq
  integer :: n, m, s, k
  do k = 1, n
    s = s + 1
  end do
  do k = k, m
    s = s + 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_by_next_loop(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_by_next_loopEk")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[LB:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) = (%[[LB]] : i32)

! The caller reads the function result.
function result_variable(n)
  !$acc routine seq
  integer :: n, result_variable
  do result_variable = 1, n
  end do
end function

! CHECK-LABEL: func.func @_QPresult_variable(
! CHECK: %[[RES:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFresult_variableEresult_variable")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[RES]]#0 : !fir.ref<i32>
! CHECK: %[[RES_AFTER:.*]] = fir.load %[[RES]]#0 : !fir.ref<i32>
! CHECK: return %[[RES_AFTER]] : i32

! The variable is only used in the loop: nothing is assigned after it.
subroutine not_used_after(a, n)
  !$acc routine seq
  integer :: n, k
  real :: a(n)
  do k = 1, n
    a(k) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPnot_used_after(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: return

! Another loop on the variable and a private clause do not use its value.
subroutine only_other_loops(a, b, n)
  integer :: n, k
  real :: a(n), b(n)
  !$acc parallel private(k)
  do k = 1, n
    a(k) = 0.
  end do
  do k = 1, n
    b(k) = a(k)
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPonly_other_loops(
! CHECK: acc.parallel private(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: return

! An I/O implied DO on the variable gives it new values.
subroutine only_io_implied_do(a, n)
  !$acc routine seq
  integer :: n, k
  real :: a(n)
  do k = 1, n
    a(k) = k
  end do
  print *, (a(k), k = 1, n)
end subroutine

! CHECK-LABEL: func.func @_QPonly_io_implied_do(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: fir.call @_FortranAioBeginExternalListOutput

! The variable is private to the thread that executes the loop in the compute
! construct: a use after the construct does not see its value.
subroutine used_after_construct(a, n, r)
  integer :: n, k, r
  real :: a(n)
  !$acc kernels
  do k = 1, n
    a(k) = 0.
  end do
  !$acc end kernels
  r = k
end subroutine

! CHECK-LABEL: func.func @_QPused_after_construct(
! CHECK: acc.kernels
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: return

! A loop with a loop directive is not changed.
subroutine explicit_loop_directive(n, r)
  integer :: n, k, r
  !$acc serial copyout(r)
  !$acc loop seq
  do k = 1, n
  end do
  r = k
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_loop_directive(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: return
