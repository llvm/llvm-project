! RUN: bbc -emit-fir %s -o - | FileCheck %s
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
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_4:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_4]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_7:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:    fir.call @_FortranAAssign
! CHECK:    fir.result %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_20]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_22:.*]] = arith.constant false
! CHECK:  %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_22]] : i1
! CHECK:  %[[VAL_24:.*]] = fir.shape %[[VAL_21]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_25:.*]] = fir.embox %[[VAL_3]](%[[VAL_24]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_25]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_23]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_3]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_to_contiguous(x)
  real, contiguous :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_3]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_1]](%[[VAL_5]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_6]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK-NEXT:  return

subroutine test_assumed_shape_opt_to_contiguous(x)
  real, optional :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_opt_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_4:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_4]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_7:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:    fir.call @_FortranAAssign
! CHECK:    fir.result %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_20]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_22:.*]] = arith.constant false
! CHECK:  %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_22]] : i1
! CHECK:  %[[VAL_24:.*]] = fir.shape %[[VAL_21]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_25:.*]] = fir.embox %[[VAL_3]](%[[VAL_24]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_25]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_23]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_3]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_opt_to_contiguous(x)
  real, optional, contiguous :: x(:)
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_opt_to_contiguous(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  fir.call @_QPtakes_contiguous(%[[VAL_0]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK-NEXT:  return


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
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_4:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_4]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_7:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:    fir.call @_FortranAAssign
! CHECK:    fir.result %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_20]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_22:.*]] = arith.constant false
! CHECK:  %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_22]] : i1
! CHECK:  %[[VAL_24:.*]] = fir.shape %[[VAL_21]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_25:.*]] = fir.embox %[[VAL_3]](%[[VAL_24]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_25]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_23]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_3]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_to_contiguous_opt(x)
  real, contiguous :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_3]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_1]](%[[VAL_5]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_6]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK-NEXT:  return

subroutine test_assumed_shape_opt_to_contiguous_opt(x)
  real, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_1]], %[[VAL_0]], %[[VAL_5]] : !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_7]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_1]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_10:.*]] = fir.if %[[VAL_8]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:      %[[VAL_11:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:      fir.result %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    } else {
! CHECK:      %[[VAL_14:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:      fir.call @_FortranAAssign
! CHECK:      fir.result %[[VAL_14]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_28:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_28]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_29]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_31:.*]] = arith.constant false
! CHECK:  %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_31]] : i1
! CHECK:  %[[VAL_33:.*]] = arith.andi %[[VAL_1]], %[[VAL_32]] : i1
! CHECK:  %[[VAL_34:.*]] = fir.shape %[[VAL_30]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_35:.*]] = fir.embox %[[VAL_9]](%[[VAL_34]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_37:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_38:.*]] = arith.select %[[VAL_1]], %[[VAL_35]], %[[VAL_37]] : !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_38]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_33]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_assumed_shape_contiguous_opt_to_contiguous_opt(x)
  real, contiguous, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_assumed_shape_contiguous_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_0]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK-NEXT:  return

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
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_9]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_11:.*]] = fir.if %[[VAL_5]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_12:.*]] = fir.if %[[VAL_10]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:      %[[VAL_13:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:      fir.result %[[VAL_13]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    } else {
! CHECK:      %[[VAL_16:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:      fir.call @_FortranAAssign
! CHECK:      fir.result %[[VAL_16]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_12]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_31:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_31]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_32:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_33:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_32]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_34:.*]] = arith.constant false
! CHECK:  %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_34]] : i1
! CHECK:  %[[VAL_36:.*]] = arith.andi %[[VAL_5]], %[[VAL_35]] : i1
! CHECK:  %[[VAL_37:.*]] = fir.shape_shift %[[VAL_8]]#0, %[[VAL_33]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_38:.*]] = fir.embox %[[VAL_11]](%[[VAL_37]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_40:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_41:.*]] = arith.select %[[VAL_5]], %[[VAL_38]], %[[VAL_40]] : !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_41]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_36]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_pointer_contiguous_to_contiguous_opt(x)
  real, pointer, contiguous :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_contiguous_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_11:.*]] = fir.shape_shift %[[VAL_9]]#0, %[[VAL_9]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_12:.*]] = fir.embox %[[VAL_10]](%[[VAL_11]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_13:.*]] = arith.select %[[VAL_5]], %[[VAL_12]], %[[VAL_6]] : !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_13]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK-NEXT:  return

subroutine test_pointer_opt_to_contiguous_opt(x)
  real, pointer, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
! CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_9]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_11:.*]] = fir.if %[[VAL_5]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_12:.*]] = fir.if %[[VAL_10]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:      %[[VAL_13:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:      fir.result %[[VAL_13]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    } else {
! CHECK:      %[[VAL_16:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK:      fir.call @_FortranAAssign
! CHECK:      fir.result %[[VAL_16]] : !fir.heap<!fir.array<?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_12]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_31:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_31]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  %[[VAL_32:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_33:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_32]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_34:.*]] = arith.constant false
! CHECK:  %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_34]] : i1
! CHECK:  %[[VAL_36:.*]] = arith.andi %[[VAL_5]], %[[VAL_35]] : i1
! CHECK:  %[[VAL_37:.*]] = fir.shape_shift %[[VAL_8]]#0, %[[VAL_33]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_38:.*]] = fir.embox %[[VAL_11]](%[[VAL_37]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_40:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_41:.*]] = arith.select %[[VAL_5]], %[[VAL_38]], %[[VAL_40]] : !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_optional(%[[VAL_41]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  fir.if %[[VAL_36]] {
! CHECK:    fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  return
! CHECK:}

subroutine test_pointer_contiguous_opt_to_contiguous_opt(x)
  real, pointer, contiguous, optional :: x(:)
  call takes_contiguous_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QMtestsPtest_pointer_contiguous_opt_to_contiguous_opt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_11:.*]] = fir.shape_shift %[[VAL_9]]#0, %[[VAL_9]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_12:.*]] = fir.embox %[[VAL_10]](%[[VAL_11]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_13:.*]] = arith.select %[[VAL_5]], %[[VAL_12]], %[[VAL_6]] : !fir.box<!fir.array<?xf32>>
! CHECK-NEXT:  fir.call @_QPtakes_contiguous_optional(%[[VAL_13]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  return
end module
