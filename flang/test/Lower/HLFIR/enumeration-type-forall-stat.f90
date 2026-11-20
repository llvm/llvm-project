! Test lowering of NEXT/PREVIOUS with STAT= inside FORALL. The STAT= variable
! is defined inside the RHS region evaluation.
! RUN: %flang_fc1 -emit-hlfir -o - %s 2>&1 | FileCheck %s
module enum_forall_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

subroutine test_forall_next_stat(a, n, st)
  use enum_forall_mod
  type(color), intent(in) :: a(3)
  type(color), intent(out) :: n(3)
  integer, intent(out) :: st(3)
  integer :: i
  forall (i = 1:3)
    n(i) = next(a(i), stat=st(i))
  end forall
end subroutine
! CHECK-LABEL: func.func @_QPtest_forall_next_stat(
! CHECK: %[[ST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_forall_next_statEst")
! CHECK: hlfir.forall
! CHECK: hlfir.region_assign {
! CHECK: %[[STI:.*]] = hlfir.designate %[[ST]]#0 (%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[STAT:.*]] = arith.select %{{.*}}, %c112_i32, %c0_i32 : i32
! CHECK: hlfir.assign %[[STAT]] to %[[STI]] : i32, !fir.ref<i32>
! CHECK: hlfir.yield %{{.*}} : !hlfir.expr<!fir.type<_QMenum_forall_modTcolor{__ordinal:i32}>> cleanup {
! CHECK: } to {
