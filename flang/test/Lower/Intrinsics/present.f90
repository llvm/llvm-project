! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPpresent_test(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?xi32>>
subroutine present_test(a)
  integer, optional :: a(:)

  if (present(a)) print *,a
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %{{.*}} = fir.is_present %[[ADECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> i1
end subroutine
