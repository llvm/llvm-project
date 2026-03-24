! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPminval_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32
integer function minval_test(a)
integer :: a(:)
! CHECK: %[[a:.*]]:2 = hlfir.declare %[[arg0]] {{.*}} {uniq_name = "_QFminval_testEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: %[[res:.*]] = hlfir.minval %[[a]]#0 {{.*}} : (!fir.box<!fir.array<?xi32>>) -> i32
minval_test = minval(a)
end function

! CHECK-LABEL: func @_QPminval_test2(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1>>{{.*}}, %[[arg1:.*]]: index{{.*}}, %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.char<1>>>{{.*}}) -> !fir.boxchar<1>
character function minval_test2(a)
character :: a(:)
! CHECK: %[[a:.*]]:2 = hlfir.declare %[[arg2]] typeparams {{.*}} {uniq_name = "_QFminval_test2Ea"} : (!fir.box<!fir.array<?x!fir.char<1>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1>>>, !fir.box<!fir.array<?x!fir.char<1>>>)
! CHECK: %[[res:.*]] = hlfir.minval %[[a]]#0 {{.*}} : (!fir.box<!fir.array<?x!fir.char<1>>>) -> !hlfir.expr<!fir.char<1>>
minval_test2 = minval(a)
end function

! CHECK-LABEL: func @_QPminval_test3(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}})
subroutine minval_test3(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK: %[[a:.*]]:2 = hlfir.declare %[[arg0]] {{.*}} {uniq_name = "_QFminval_test3Ea"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK: %[[r:.*]]:2 = hlfir.declare %[[arg1]] {{.*}} {uniq_name = "_QFminval_test3Er"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: %[[c2:.*]] = arith.constant 2 : i32
! CHECK: %[[res:.*]] = hlfir.minval %[[a]]#0 dim %[[c2]] {{.*}} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK: hlfir.assign %[[res]] to %[[r]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK: hlfir.destroy %[[res]] : !hlfir.expr<?xi32>
r = minval(a,dim=2)
end subroutine

! CHECK-LABEL: func @_QPtest_minval_optional_scalar_mask(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>
subroutine test_minval_optional_scalar_mask(mask, array)
integer :: array(:)
logical, optional :: mask
print *, minval(array, mask)
! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_minval_optional_scalar_maskEmask"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[present:.*]] = fir.is_present %[[mask_decl]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK: %[[box:.*]] = fir.embox %[[mask_decl]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.logical<4>>
! CHECK: %[[mask:.*]] = arith.select %[[present]], %[[box]], %[[absent]] : !fir.box<!fir.logical<4>>
! CHECK: hlfir.minval {{.*}} mask %[[mask]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.logical<4>>) -> i32
end subroutine

! CHECK-LABEL: func @_QPtest_minval_optional_array_mask(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine test_minval_optional_array_mask(mask, array)
integer :: array(:)
logical, optional :: mask(:)
print *, minval(array, mask)
! CHECK: %[[mask_decl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_minval_optional_array_maskEmask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK: %[[present:.*]] = fir.is_present %[[mask_decl]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK: %[[absent:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK: %[[mask:.*]] = arith.select %[[present]], %[[mask_decl]]#1, %[[absent]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK: hlfir.minval {{.*}} mask %[[mask]] {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
end subroutine
