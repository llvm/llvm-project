! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmaxval_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function maxval_test(a)
integer :: a(:)
! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmaxval_testEa"}
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK-DAG:  %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFmaxval_testEmaxval_test"}
maxval_test = maxval(a)
! CHECK:  %[[MAXVAL:.*]] = hlfir.maxval %[[A]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:  hlfir.assign %[[MAXVAL]] to %[[RES]]#0 : i32, !fir.ref<i32>
end function

! CHECK-LABEL: func @_QPmaxval_test2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.char<1>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.char<1>>> {fir.bindc_name = "a"}) -> !fir.boxchar<1> {
character function maxval_test2(a)
character :: a(:)
! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[ARG2]] typeparams %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmaxval_test2Ea"}
! CHECK-DAG:  %[[RES:.*]]:2 = hlfir.declare %[[ARG0]] typeparams %{{.*}} {uniq_name = "_QFmaxval_test2Emaxval_test2"}
maxval_test2 = maxval(a)
! CHECK:  %[[MAXVAL:.*]] = hlfir.maxval %[[A]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1>>>) -> !hlfir.expr<!fir.char<1>>
! CHECK:  hlfir.assign %[[MAXVAL]] to %[[RES]]#0 : !hlfir.expr<!fir.char<1>>, !fir.ref<!fir.char<1>>
! CHECK:  hlfir.destroy %[[MAXVAL]] : !hlfir.expr<!fir.char<1>>
end function

! CHECK-LABEL: func @_QPmaxval_test3(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine maxval_test3(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmaxval_test3Ea"}
! CHECK-DAG:  %[[R:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmaxval_test3Er"}
! CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : i32
r = maxval(a,dim=2)
! CHECK:  %[[MAXVAL:.*]] = hlfir.maxval %[[A]]#0 dim %[[C2]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK:  hlfir.assign %[[MAXVAL]] to %[[R]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:  hlfir.destroy %[[MAXVAL]] : !hlfir.expr<?xi32>
end subroutine

! CHECK-LABEL: func @_QPtest_maxval_optional_scalar_mask(
! CHECK-SAME:  %[[VAL_0:[^:]+]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask", fir.optional}
subroutine test_maxval_optional_scalar_mask(mask, array)
integer :: array(:)
logical, optional :: mask
print *, maxval(array, mask)
! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<optional>,
! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[MASK]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[MASK]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.logical<4>>
! CHECK:  %[[MASK_SEL:.*]] = arith.select %[[IS_PRESENT]], %[[EMBOX]], %[[ABSENT]] : !fir.box<!fir.logical<4>>
! CHECK:  hlfir.maxval %{{.*}} mask %[[MASK_SEL]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.logical<4>>) -> i32
end subroutine

! CHECK-LABEL: func @_QPtest_maxval_optional_array_mask(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "mask", fir.optional}
subroutine test_maxval_optional_array_mask(mask, array)
integer :: array(:)
logical, optional :: mask(:)
print *, maxval(array, mask)
! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<optional>,
! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[MASK]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[MASK_SEL:.*]] = arith.select %[[IS_PRESENT]], %[[MASK]]#1, %[[ABSENT]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  hlfir.maxval %{{.*}} mask %[[MASK_SEL]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
end subroutine
