! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPishftc_test(
! CHECK-SAME:  %[[I_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:  %[[J_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "j"},
! CHECK-SAME:  %[[K_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "k"}) -> i32 {
function ishftc_test(i, j, k)
! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ARG]]
! CHECK-DAG: %[[result:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFishftc_testEishftc_test"}
! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ARG]]
! CHECK-DAG: %[[K:.*]]:2 = hlfir.declare %[[K_ARG]]
! CHECK-DAG: %[[i:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[j:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[k:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 32 : i32
! CHECK-DAG: %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK-DAG: %[[VAL_9:.*]] = arith.constant -1 : i32
! CHECK-DAG: %[[VAL_10:.*]] = arith.constant 31 : i32
! CHECK: %[[VAL_11:.*]] = arith.shrsi %[[j]], %[[VAL_10]] : i32
! CHECK: %[[VAL_12:.*]] = arith.xori %[[j]], %[[VAL_11]] : i32
! CHECK: %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_11]] : i32
! CHECK: %[[VAL_14:.*]] = arith.subi %[[k]], %[[VAL_13]] : i32
! CHECK: %[[VAL_15:.*]] = arith.cmpi eq, %[[j]], %[[VAL_8]] : i32
! CHECK: %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_13]], %[[k]] : i32
! CHECK: %[[VAL_17:.*]] = arith.ori %[[VAL_15]], %[[VAL_16]] : i1
! CHECK: %[[VAL_18:.*]] = arith.cmpi sgt, %[[j]], %[[VAL_8]] : i32
! CHECK: %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_13]], %[[VAL_14]] : i32
! CHECK: %[[VAL_20:.*]] = arith.select %[[VAL_18]], %[[VAL_14]], %[[VAL_13]] : i32
! CHECK: %[[VAL_21:.*]] = arith.cmpi ne, %[[k]], %[[VAL_7]] : i32
! CHECK: %[[VAL_22:.*]] = arith.shrui %[[i]], %[[k]] : i32
! CHECK: %[[VAL_23:.*]] = arith.shli %[[VAL_22]], %[[k]] : i32
! CHECK: %[[VAL_24:.*]] = arith.select %[[VAL_21]], %[[VAL_23]], %[[VAL_8]] : i32
! CHECK: %[[VAL_25:.*]] = arith.subi %[[VAL_7]], %[[VAL_19]] : i32
! CHECK: %[[VAL_26:.*]] = arith.shrui %[[VAL_9]], %[[VAL_25]] : i32
! CHECK: %[[VAL_27:.*]] = arith.shrui %[[i]], %[[VAL_20]] : i32
! CHECK: %[[VAL_28:.*]] = arith.andi %[[VAL_27]], %[[VAL_26]] : i32
! CHECK: %[[VAL_29:.*]] = arith.subi %[[VAL_7]], %[[VAL_20]] : i32
! CHECK: %[[VAL_30:.*]] = arith.shrui %[[VAL_9]], %[[VAL_29]] : i32
! CHECK: %[[VAL_31:.*]] = arith.andi %[[i]], %[[VAL_30]] : i32
! CHECK: %[[VAL_32:.*]] = arith.shli %[[VAL_31]], %[[VAL_19]] : i32
! CHECK: %[[VAL_33:.*]] = arith.ori %[[VAL_24]], %[[VAL_28]] : i32
! CHECK: %[[VAL_34:.*]] = arith.ori %[[VAL_33]], %[[VAL_32]] : i32
! CHECK: %[[VAL_35:.*]] = arith.select %[[VAL_17]], %[[i]], %[[VAL_34]] : i32
! CHECK: hlfir.assign %[[VAL_35]] to %[[result]]#0 : i32, !fir.ref<i32>
! CHECK: %[[VAL_36:.*]] = fir.load %[[result]]#0 : !fir.ref<i32>
! CHECK: return %[[VAL_36]] : i32
  ishftc_test = ishftc(i, j, k)
end

  ! Test cases where the size argument presence can only be know at runtime
  module test_ishftc
  contains
  ! CHECK-LABEL: func.func @_QMtest_ishftcPdyn_optional_scalar(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"},
  ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "size", fir.optional}) {
  subroutine dyn_optional_scalar(i, shift, size)
    integer, optional :: size
    integer :: i, shift
    print *, ishftc(i, shift, size)
    ! CHECK:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
    ! CHECK:  %[[SHIFT:.*]]:2 = hlfir.declare %[[VAL_1]]
    ! CHECK:  %[[SIZE:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {fortran_attrs = #fir.var_attrs<optional>
    ! CHECK:  %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
    ! CHECK:  %[[SHIFT_VAL:.*]] = fir.load %[[SHIFT]]#0 : !fir.ref<i32>
    ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[SIZE]]#0 : (!fir.ref<i32>) -> i1
    ! CHECK:  %[[SIZE_VAL:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
    ! CHECK:    %[[LOADED:.*]] = fir.load %[[SIZE]]#0 : !fir.ref<i32>
    ! CHECK:    fir.result %[[LOADED]] : i32
    ! CHECK:  } else {
    ! CHECK:    %[[DEFAULT:.*]] = arith.constant 32 : i32
    ! CHECK:    fir.result %[[DEFAULT]] : i32
    ! CHECK:  }
    ! CHECK:  %[[c31:.*]] = arith.constant 31 : i32
    ! CHECK:  %[[VAL_18:.*]] = arith.shrsi %[[SHIFT_VAL]], %[[c31]] : i32
    ! CHECK:  %[[VAL_19:.*]] = arith.xori %[[SHIFT_VAL]], %[[VAL_18]] : i32
    ! CHECK:  %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i32
    ! CHECK:  %[[VAL_21:.*]] = arith.subi %[[SIZE_VAL]], %[[VAL_20]] : i32
    ! ... as in non optional case
  end subroutine

  ! CHECK-LABEL: func.func @_QMtest_ishftcPdyn_optional_array_scalar(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "shift"},
  ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "size", fir.optional}) {
  subroutine dyn_optional_array_scalar(i, shift, size)
    integer, optional :: size
    integer :: i(:), shift(:)
  ! CHECK:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
  ! CHECK:  %[[SHIFT:.*]]:2 = hlfir.declare %[[VAL_1]]
  ! CHECK:  %[[SIZE:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {fortran_attrs = #fir.var_attrs<optional>
  ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[SIZE]]#0 : (!fir.ref<i32>) -> i1
  ! CHECK:  %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ! CHECK:  ^bb0(%[[IDX:.*]]: index):
  ! CHECK:    %[[I_ELEM:.*]] = hlfir.designate %[[I]]#0 (%[[IDX]])
  ! CHECK:    %[[I_VAL:.*]] = fir.load %[[I_ELEM]] : !fir.ref<i32>
  ! CHECK:    %[[S_ELEM:.*]] = hlfir.designate %[[SHIFT]]#0 (%[[IDX]])
  ! CHECK:    %[[S_VAL:.*]] = fir.load %[[S_ELEM]] : !fir.ref<i32>
  ! CHECK:    %[[SIZE_VAL:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
  ! CHECK:      %[[LOADED:.*]] = fir.load %[[SIZE]]#0 : !fir.ref<i32>
  ! CHECK:      fir.result %[[LOADED]] : i32
  ! CHECK:    } else {
  ! CHECK:      %[[DEFAULT:.*]] = arith.constant 32 : i32
  ! CHECK:      fir.result %[[DEFAULT]] : i32
  ! CHECK:    }
  ! ... as in non optional case
  ! CHECK:  }
    print *, ishftc(i, shift, size)
  end subroutine

  ! CHECK-LABEL: func.func @_QMtest_ishftcPdyn_optional_array(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "shift"},
  ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "size", fir.optional}) {
  subroutine dyn_optional_array(i, shift, size)
    integer, optional :: size(:)
    integer :: i(:), shift(:)
  ! CHECK:  %[[I:.*]]:2 = hlfir.declare %[[VAL_0]]
  ! CHECK:  %[[SHIFT:.*]]:2 = hlfir.declare %[[VAL_1]]
  ! CHECK:  %[[SIZE:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {fortran_attrs = #fir.var_attrs<optional>
  ! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[SIZE]]#0 : (!fir.box<!fir.array<?xi32>>) -> i1
  ! CHECK:  %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ! CHECK:  ^bb0(%[[IDX:.*]]: index):
  ! CHECK:    %[[I_ELEM:.*]] = hlfir.designate %[[I]]#0 (%[[IDX]])
  ! CHECK:    %[[I_VAL:.*]] = fir.load %[[I_ELEM]] : !fir.ref<i32>
  ! CHECK:    %[[S_ELEM:.*]] = hlfir.designate %[[SHIFT]]#0 (%[[IDX]])
  ! CHECK:    %[[S_VAL:.*]] = fir.load %[[S_ELEM]] : !fir.ref<i32>
  ! CHECK:    %[[SIZE_VAL:.*]] = fir.if %[[IS_PRESENT]] -> (i32) {
  ! CHECK:      %[[SIZE_ELEM:.*]] = hlfir.designate %[[SIZE]]#0 (%[[IDX]])
  ! CHECK:      %[[LOADED:.*]] = fir.load %[[SIZE_ELEM]] : !fir.ref<i32>
  ! CHECK:      fir.result %[[LOADED]] : i32
  ! CHECK:    } else {
  ! CHECK:      %[[DEFAULT:.*]] = arith.constant 32 : i32
  ! CHECK:      fir.result %[[DEFAULT]] : i32
  ! CHECK:    }
  ! ... as in non optional case
  ! CHECK:    }
    print *, ishftc(i, shift, size)
  end subroutine
  end module

    use test_ishftc
    integer :: i(4) = [333, 334, 335, 336]
    integer :: shift(4) = [2, 1, -1, -2]
    integer :: size(4) = [2, 4, 8, 16]
    call dyn_optional_scalar(i(1), shift(1))
    call dyn_optional_scalar(i(1), shift(1), size(1))

    call dyn_optional_array_scalar(i, shift)
    call dyn_optional_array_scalar(i, shift, size(1))

    call dyn_optional_array(i, shift)
    call dyn_optional_array(i, shift, size)
  end
