! RUN: %flang_fc1 -emit-hlfir -mmlir --unsafe-cray-pointers -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPcray_pointer_loop()
subroutine cray_pointer_loop
  integer :: g, g5
  integer(8) :: lm
  pointer(lm, gpl6l)

  ! CHECK: %[[G_REF:.*]] = fir.alloca i32 {bindc_name = "g"
  ! CHECK: %[[G:.*]]:2 = hlfir.declare %[[G_REF]]
  lm = loc(g)
  g5 = 0

  ! CHECK: %[[LB:.*]] = arith.constant -2 : i32
  ! CHECK: %[[UB:.*]] = arith.constant -7 : i32
  ! CHECK: %[[STEP:.*]] = arith.constant -2 : i32
  ! CHECK: fir.store %[[LB]] to %[[G]]#0 : !fir.ref<i32>
  ! CHECK: fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] : i32 {
  ! CHECK-NOT: fir.store %[[IV]] to %[[G]]#0
  do g = -2, -7, -2
    g5 = g5 + 5
    gpl6l = g - 3
  ! CHECK: %[[UPDATED:.*]] = fir.load %[[G]]#0 : !fir.ref<i32>
  ! CHECK: %[[NEXT:.*]] = arith.addi %[[UPDATED]], %[[STEP]] overflow<nsw> : i32
  ! CHECK: fir.store %[[NEXT]] to %[[G]]#0 : !fir.ref<i32>
  ! CHECK: fir.result
  ! CHECK: }
  end do
end subroutine

! CHECK-LABEL: func.func @_QPpointer_alias_loop()
subroutine pointer_alias_loop
  integer, target :: i
  integer, pointer :: p

  p => i
  ! CHECK: %[[I_REF:.*]] = fir.alloca i32 {bindc_name = "i"
  ! CHECK: %[[I:.*]]:2 = hlfir.declare %[[I_REF]]
  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK: %[[UB:.*]] = arith.constant 3 : i32
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: fir.store %[[LB]] to %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] : i32 {
  ! CHECK-NOT: fir.store %[[IV]] to %[[I]]#0
  do i = 1, 3
    p = i + 1
  ! CHECK: %[[UPDATED:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK: %[[NEXT:.*]] = arith.addi %[[UPDATED]], %[[STEP]] overflow<nsw> : i32
  ! CHECK: fir.store %[[NEXT]] to %[[I]]#0 : !fir.ref<i32>
  ! CHECK: fir.result
  ! CHECK: }
  end do
end subroutine
