! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPany_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {{.*}}) -> !fir.logical<4>
logical function any_test(mask)
logical :: mask(:)
! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
! CHECK: %[[res:.*]] = hlfir.any %[[mask_decl]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
! CHECK: hlfir.assign %[[res]] to %{{.*}} : !fir.logical<4>, !fir.ref<!fir.logical<4>>
any_test = any(mask)
end function any_test

! CHECK-LABEL: func.func @_QPany_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {{.*}})
subroutine any_test2(mask, d, rslt)
logical :: mask(:,:)
integer :: d
logical :: rslt(:)
! CHECK: %[[d_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}
! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
! CHECK: %[[rslt_decl:.*]]:2 = hlfir.declare %[[arg2]] {{.*}}
! CHECK: %[[d_val:.*]] = fir.load %[[d_decl]]#0 : !fir.ref<i32>
! CHECK: %[[res:.*]] = hlfir.any %[[mask_decl]]#0 dim %[[d_val]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
! CHECK: hlfir.assign %[[res]] to %[[rslt_decl]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK: hlfir.destroy %[[res]] : !hlfir.expr<?x!fir.logical<4>>
rslt = any(mask, d)
end subroutine
