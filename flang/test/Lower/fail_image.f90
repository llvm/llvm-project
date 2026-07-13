! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPfail_image_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "fail"}) {
subroutine fail_image_test(fail)
  logical :: fail
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.logical<4>) -> i1
! CHECK:  cf.cond_br %[[VAL_3]], ^[[BB1:.*]], ^[[BB2:.*]]
! CHECK: ^[[BB1]]:
  if (fail) then
! CHECK: fir.call @_FortranAFailImageStatement() {{.*}}: () -> ()
! CHECK-NEXT:  fir.unreachable
   FAIL IMAGE
  end if
! CHECK: ^[[BB2]]:
! CHECK-NEXT:  cf.br ^[[BB3:.*]]
! CHECK-NEXT: ^[[BB3]]:
! CHECK-NEXT:  return
  return
end subroutine
! CHECK-LABEL: func.func private @_FortranAFailImageStatement() attributes {fir.runtime}
