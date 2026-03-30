! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPminloc_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine minloc_test(arr,res)
    integer :: arr(:)
    integer :: res(:)
  ! CHECK-DAG: %[[ARR:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFminloc_testEarr"}
  ! CHECK-DAG: %[[RES:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFminloc_testEres"}
    res = minloc(arr)
  ! CHECK: %[[MINLOC:.*]] = hlfir.minloc %[[ARR]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<1xi32>
  ! CHECK: hlfir.assign %[[MINLOC]] to %[[RES]]#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
  ! CHECK: hlfir.destroy %[[MINLOC]] : !hlfir.expr<1xi32>
  end subroutine

  ! CHECK-LABEL: func @_QPminloc_test2(
  ! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[ARG2:.*]]: !fir.ref<i32>{{.*}}) {
  subroutine minloc_test2(arr,res,d)
    integer :: arr(:)
    integer :: res(:)
    integer :: d
  ! CHECK-DAG:  %[[ARR:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFminloc_test2Earr"}
  ! CHECK-DAG:  %[[D:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{.*}} arg 3 {uniq_name = "_QFminloc_test2Ed"}
  ! CHECK-DAG:  %[[RES:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFminloc_test2Eres"}
    res = minloc(arr, dim=d)
  ! CHECK:  %[[D_VAL:.*]] = fir.load %[[D]]#0 : !fir.ref<i32>
  ! CHECK:  %[[MINLOC:.*]] = hlfir.minloc %[[ARR]]#0 dim %[[D_VAL]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, i32) -> i32
  ! CHECK:  hlfir.assign %[[MINLOC]] to %[[RES]]#0 : i32, !fir.box<!fir.array<?xi32>>
  end subroutine

  ! CHECK-LABEL: func @_QPtest_minloc_optional_scalar_mask(
  ! CHECK-SAME:  %[[VAL_0:[^:]+]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask", fir.optional}
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "back", fir.optional}
  subroutine test_minloc_optional_scalar_mask(mask, back, array)
    integer :: array(:)
    logical, optional :: mask
    logical, optional :: back
    print *, minloc(array, mask=mask, back=back)
  ! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<optional>,
  ! CHECK-DAG:  %[[BACK:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} arg 2 {fortran_attrs = #fir.var_attrs<optional>,
  ! CHECK:  %[[IS_PRESENT_MASK:.*]] = fir.is_present %[[MASK]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[IS_PRESENT_BACK:.*]] = fir.is_present %[[BACK]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[EMBOX:.*]] = fir.embox %[[MASK]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.logical<4>>
  ! CHECK:  %[[MASK_SEL:.*]] = arith.select %[[IS_PRESENT_MASK]], %[[EMBOX]], %[[ABSENT]] : !fir.box<!fir.logical<4>>
  ! CHECK:  %[[BACK_VAL:.*]] = fir.if %[[IS_PRESENT_BACK]] -> (!fir.logical<4>)
  ! CHECK:  hlfir.minloc %{{.*}} mask %[[MASK_SEL]] back %[[BACK_VAL]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.logical<4>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
  end subroutine

  ! CHECK-LABEL: func @_QPtest_minloc_optional_array_mask(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "mask", fir.optional}
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "back", fir.optional}
  subroutine test_minloc_optional_array_mask(mask, back, array)
    integer :: array(:)
    logical, optional :: mask(:)
    logical, optional :: back
    print *, minloc(array, mask=mask, back=back)
  ! CHECK-DAG:  %[[MASK:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<optional>,
  ! CHECK-DAG:  %[[BACK:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} arg 2 {fortran_attrs = #fir.var_attrs<optional>,
  ! CHECK:  %[[IS_PRESENT_MASK:.*]] = fir.is_present %[[MASK]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
  ! CHECK:  %[[IS_PRESENT_BACK:.*]] = fir.is_present %[[BACK]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
  ! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
  ! CHECK:  %[[MASK_SEL:.*]] = arith.select %[[IS_PRESENT_MASK]], %[[MASK]]#1, %[[ABSENT]] : !fir.box<!fir.array<?x!fir.logical<4>>>
  ! CHECK:  %[[BACK_VAL:.*]] = fir.if %[[IS_PRESENT_BACK]] -> (!fir.logical<4>)
  ! CHECK:  hlfir.minloc %{{.*}} mask %[[MASK_SEL]] back %[[BACK_VAL]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
  end subroutine
