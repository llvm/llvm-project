! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
module tests
interface
  subroutine takes_contiguous(a)
    real, contiguous :: a(:)
  end subroutine
  subroutine takes_contiguous_optional(a)
    real, contiguous, optional :: a(:)
  end subroutine
end interface

contains

! -----------------------------------------------------------------------------
!     Test passing assumed shapes to contiguous assumed shapes
! -----------------------------------------------------------------------------
! Base case.

subroutine test_assumed_shape_to_contiguous(x)
  real :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_to_contiguousEx"{{.*}}
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_2]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_3]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_3]]#1 to %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_to_contiguous(x)
  real, contiguous :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]]:3 = fir.box_dims %[[VAL_0]], %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_3:.*]] = fir.shape_shift %c1, %[[VAL_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_contiguous_to_contiguousEx"{{.*}}
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_4]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_opt_to_contiguous(x)
  real, optional :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_opt_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_opt_to_contiguousEx"{{.*}}
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_2]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_3]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_3]]#1 to %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_opt_to_contiguous(x)
  real, optional, contiguous :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_opt_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_contiguous_opt_to_contiguousEx"{{.*}}
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_1]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

! -----------------------------------------------------------------------------
!     Test passing assumed shapes to contiguous optional assumed shapes
! -----------------------------------------------------------------------------
! The copy-in/out must take into account the actual argument presence (which may
! not be known until runtime).

subroutine test_assumed_shape_to_contiguous_opt(x)
  real :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_2]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_3]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_3]]#1 to %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_to_contiguous_opt(x)
  real, contiguous :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]]:3 = fir.box_dims %[[VAL_0]], %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_3:.*]] = fir.shape_shift %c1, %[[VAL_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_contiguous_to_contiguous_optEx"{{.*}}
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_4]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_opt_to_contiguous_opt(x)
  real, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_opt_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_3:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_4:.*]]:3 = fir.if %[[VAL_3]] -> (!fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.copy_in %[[VAL_2]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:    fir.result %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_2]]#0 : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_6:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    %[[VAL_7:.*]] = arith.constant false
! CHECK:    %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_4]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_4]]#1 to %[[VAL_4]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_opt_to_contiguous_opt(x)
  real, contiguous, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_assumed_shape_contiguous_opt_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.box<!fir.array<?xf32>>) {
! CHECK:    fir.result %[[VAL_1]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_4:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_4]] : !fir.box<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_3]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

! -----------------------------------------------------------------------------
!     Test passing pointers to contiguous optional assumed shapes
! -----------------------------------------------------------------------------
! This case is interesting because pointers may be non contiguous, and also because
! a pointer passed to an optional assumed shape dummy is present if and only if the
! pointer is associated (regardless of the pointer optionality).

subroutine test_pointer_to_contiguous_opt(x)
  real, pointer :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_pointer_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %c0_i64 : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
! CHECK:    %[[VAL_8:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.copy_in %[[VAL_8]] to %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
! CHECK:    %[[VAL_10:.*]] = fir.rebox %[[VAL_9]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_10]], %[[VAL_9]]#1, %[[VAL_8]] : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    %[[VAL_12:.*]] = arith.constant false
! CHECK:    %[[VAL_13:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:    fir.result %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_7]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_7]]#1 to %[[VAL_7]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_pointer_contiguous_to_contiguous_opt(x)
  real, pointer, contiguous :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_contiguous_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_pointer_contiguous_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_4]], %c0_i64 : i64
! CHECK:  %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.box<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_7:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:    %[[VAL_8:.*]] = fir.rebox %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_8]] : !fir.box<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_9:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_9]] : !fir.box<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_6]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_pointer_opt_to_contiguous_opt(x)
  real, pointer, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_pointer_opt_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %c0_i64 : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
! CHECK:    %[[VAL_8:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:    %[[VAL_9:.*]]:2 = hlfir.copy_in %[[VAL_8]] to %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
! CHECK:    %[[VAL_10:.*]] = fir.rebox %[[VAL_9]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_10]], %[[VAL_9]]#1, %[[VAL_8]] : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    %[[VAL_12:.*]] = arith.constant false
! CHECK:    %[[VAL_13:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:    fir.result %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_7]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_7]]#1 to %[[VAL_7]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()
! CHECK:  return
! CHECK:}

subroutine test_pointer_contiguous_opt_to_contiguous_opt(x)
  real, pointer, contiguous, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_contiguous_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QMtestsFtest_pointer_contiguous_opt_to_contiguous_optEx"{{.*}}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_4]], %c0_i64 : i64
! CHECK:  %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.box<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_7:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:    %[[VAL_8:.*]] = fir.rebox %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_8]] : !fir.box<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_9:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_9]] : !fir.box<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_6]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
! CHECK:}
end module
