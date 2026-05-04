! Test lowering of OPTIONAL VALUE dummy argument on caller side.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! A copy must be made if the actual is a variable (and no copy-out), but care
! has to be take if the actual argument may be absent at runtime: the copy
! must be conditional. When the allocation is dynamic, the temp allocation and
! deallocation are also conditionals.

module test
interface
  subroutine scalar(i)
    integer, optional, value :: i
  end subroutine
  subroutine dyn_char(c)
    character(*), optional, value :: c
  end subroutine
  subroutine array(i)
    integer, optional, value :: i(100)
  end subroutine
  subroutine dyn_array(i, n)
    integer(8) :: n
    integer, optional, value :: i(n)
  end subroutine
  subroutine dyn_char_array(c, n)
    integer(8) :: n
    character(*), optional, value :: c(n)
  end subroutine
  function returns_ptr()
    integer, pointer :: returns_ptr
  end function
end interface
contains

! CHECK-LABEL: func @_QMtestPtest_scalar_not_a_var() {
subroutine test_scalar_not_a_var()
  call scalar(42)
! CHECK:  %[[VAL_0:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:  fir.call @_QPscalar(%[[VAL_1]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar(i)
  integer, optional :: i
  call scalar(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]]#0 : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.ref<i32>, !fir.ref<i32>, i1) {
! CHECK:    %[[VAL_4:.*]] = hlfir.as_expr %[[VAL_0]]#0 : (!fir.ref<i32>) -> !hlfir.expr<i32>
! CHECK:    %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    fir.result %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  } else {
! CHECK:    %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[VAL_7:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[VAL_6]], %[[VAL_7]], %[[FALSE]] : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_3]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar2(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar2(i)
  integer, optional, value :: i
  call scalar(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]]#0 : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.ref<i32>, !fir.ref<i32>, i1) {
! CHECK:    %[[VAL_4:.*]] = hlfir.as_expr %[[VAL_0]]#0 : (!fir.ref<i32>) -> !hlfir.expr<i32>
! CHECK:    %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    fir.result %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  } else {
! CHECK:    %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[VAL_7:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[VAL_6]], %[[VAL_7]], %[[FALSE]] : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_3]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar3(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar3(i)
  integer, optional :: i
  ! i must be present when it appears in "()"
  call scalar((i))
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = hlfir.no_reassoc %[[VAL_2]] : i32
! CHECK:  %[[VAL_4:.*]]:3 = hlfir.associate %[[VAL_3]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:  fir.call @_QPscalar(%[[VAL_4]]#0) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar_ptr(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "i"}) {
subroutine test_scalar_ptr(i)
  integer, pointer :: i
  call scalar(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<i32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.ref<i32>, !fir.ref<i32>, i1) {
! CHECK:    %[[VAL_PLOAD:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_PADDR:.*]] = fir.box_addr %[[VAL_PLOAD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_PADDR]] : (!fir.ptr<i32>) -> !hlfir.expr<i32>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#0, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_7]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar_simple_var(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
subroutine test_scalar_simple_var(i)
  integer :: i
  call scalar(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = hlfir.as_expr %[[VAL_0]]#0 : (!fir.ref<i32>) -> !hlfir.expr<i32>
! CHECK:  %[[VAL_3:.*]]:3 = hlfir.associate %[[VAL_2]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:  fir.call @_QPscalar(%[[VAL_3]]#0) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine


! CHECK-LABEL: func @_QMtestPtest_scalar_alloc(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "i"}) {
subroutine test_scalar_alloc(i)
  integer, allocatable :: i
  call scalar(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<i32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.ref<i32>, !fir.ref<i32>, i1) {
! CHECK:    %[[VAL_HLOAD:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:    %[[VAL_HADDR:.*]] = fir.box_addr %[[VAL_HLOAD]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_HADDR]] : (!fir.heap<i32>) -> !hlfir.expr<i32>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#0, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_7]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_ptr_2() {
subroutine test_ptr_2()
  call scalar(returns_ptr())
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_2:.*]] = fir.call @_QPreturns_ptr() {{.*}}: () -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.save_result %[[VAL_2]] to %[[VAL_0]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"}
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<i32>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]]:3 = fir.if %[[VAL_8]] -> (!fir.ref<i32>, !fir.ref<i32>, i1) {
! CHECK:    %[[VAL_RLOAD:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_RADDR:.*]] = fir.box_addr %[[VAL_RLOAD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_RADDR]] : (!fir.ptr<i32>) -> !hlfir.expr<i32>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]] {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#0, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<i32>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<i32>, !fir.ref<i32>, i1
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_9]]#0) {{.*}}: (!fir.ref<i32>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_9]]#1, %[[VAL_9]]#2 : !fir.ref<i32>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "i", fir.optional}) {
subroutine test_array(i)
  integer, optional :: i(100)
  call array(i)
! CHECK:  %[[VAL_0:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_1:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_1]])
! CHECK:  %[[VAL_3:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.ref<!fir.array<100xi32>>) -> i1
! CHECK:  %[[VAL_4:.*]]:3 = fir.if %[[VAL_3]] -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>, i1) {
! CHECK:    %[[VAL_5:.*]] = hlfir.as_expr %[[VAL_2]]#0 : (!fir.ref<!fir.array<100xi32>>) -> !hlfir.expr<100xi32>
! CHECK:    %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_1]]) {adapt.valuebyref} : (!hlfir.expr<100xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>, i1)
! CHECK:    fir.result %[[VAL_6]]#0, %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>, i1
! CHECK:  } else {
! CHECK:    fir.absent !fir.ref<!fir.array<100xi32>>
! CHECK:  }
! CHECK:  fir.call @_QParray(%[[VAL_4]]#0) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array2(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME:  %[[ARG1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
subroutine test_array2(i, n)
  integer(8) :: n
  integer, optional, value :: i(n)
  call array(i)
! CHECK:  %[[VAL_N:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_N]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_I:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_SHAPE]])
! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1) {
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:    %[[VAL_CAST:.*]] = fir.convert %[[VAL_ASSOC]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:    fir.result %[[VAL_CAST]], %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<!fir.array<100xi32>>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  }
! CHECK:  fir.call @_QParray(%[[VAL_8]]#0) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_8]]#1, %[[VAL_8]]#2 : !fir.ref<!fir.array<?xi32>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_dyn_array(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME:  %[[ARG1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
subroutine test_dyn_array(i, n)
  integer(8) :: n
  integer, optional :: i(n)
  call dyn_array(i, n)
! CHECK:  %[[VAL_N:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_N]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_I:.*]]:2 = hlfir.declare %[[ARG0]](%[[VAL_SHAPE]])
! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_8:.*]]:3 = fir.if %[[VAL_7]] -> (!fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1) {
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPdyn_array(%[[VAL_8]]#0, %[[VAL_N]]#0) {{.*}}: (!fir.ref<!fir.array<?xi32>>, !fir.ref<i64>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_8]]#1, %[[VAL_8]]#2 : !fir.ref<!fir.array<?xi32>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_dyn_array_from_assumed(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME:  %[[ARG1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
subroutine test_dyn_array_from_assumed(i, n)
  integer(8) :: n
  integer, optional :: i(:)
  call dyn_array(i, n)
! CHECK:  %[[VAL_I:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_N:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1) {
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_I]]#0 : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
! CHECK:    %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_I]]#0, %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:    %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPdyn_array(%[[VAL_3]]#0, %[[VAL_N]]#0) {{.*}}: (!fir.ref<!fir.array<?xi32>>, !fir.ref<i64>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.ref<!fir.array<?xi32>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array_ptr(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "i"}) {
subroutine test_array_ptr(i)
  integer, pointer :: i(:)
  call array(i)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1) {
! CHECK:    %[[VAL_PLOAD:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_PLOAD]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !hlfir.expr<?xi32>
! CHECK:    %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_PLOAD]], %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:    %[[VAL_CAST:.*]] = fir.convert %[[VAL_ASSOC]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:    fir.result %[[VAL_CAST]], %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.ref<!fir.array<100xi32>>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.array<?xi32>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  }
! CHECK:  fir.call @_QParray(%[[VAL_7]]#0) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.array<?xi32>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c", fir.optional}) {
subroutine test_char(c)
  character(*), optional :: c
  call dyn_char(c)
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1
! CHECK:  %[[VAL_3:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.boxchar<1>) -> i1
! CHECK:  %[[VAL_4:.*]]:3 = fir.if %[[VAL_3]] -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1) {
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_2]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]] typeparams %[[VAL_1]]#1 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#0, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.boxchar<1>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.char<1,?>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPdyn_char(%[[VAL_4]]#0) {{.*}}: (!fir.boxchar<1>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_4]]#1, %[[VAL_4]]#2 : !fir.ref<!fir.char<1,?>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char_ptr(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
subroutine test_char_ptr(c)
  character(:), pointer :: c
  call dyn_char(c)
! CHECK:  %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:3 = fir.if %[[VAL_6]] -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1) {
! CHECK:    %[[VAL_PLOAD:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:    %[[VAL_PADDR:.*]] = fir.box_addr %[[VAL_PLOAD]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:    %[[VAL_LEN_BOX:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:    %[[VAL_LEN:.*]] = fir.box_elesize %[[VAL_LEN_BOX]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:    %[[VAL_EBC:.*]] = fir.emboxchar %[[VAL_PADDR]], %[[VAL_LEN]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_EBC]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]] typeparams %[[VAL_LEN]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
! CHECK:    fir.result %[[VAL_ASSOC]]#0, %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.boxchar<1>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.char<1,?>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPdyn_char(%[[VAL_7]]#0) {{.*}}: (!fir.boxchar<1>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.char<1,?>>, i1
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char_array(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.optional}) {
subroutine test_char_array(c)
  integer(8) :: n
  character(*), optional :: c(:)
  call dyn_char_array(c, n)
! CHECK:  %[[VAL_C:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:  %[[VAL_N:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QMtestFtest_char_arrayEn"}
! CHECK:  %[[VAL_3:.*]] = fir.is_present %[[VAL_C]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
! CHECK:  %[[VAL_4:.*]]:3 = fir.if %[[VAL_3]] -> (!fir.boxchar<1>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1) {
! CHECK:    %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_C]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<?x!fir.char<1,?>>
! CHECK:    %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_C]]#0, %{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_LEN:.*]] = fir.box_elesize %[[VAL_C]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_EXPR]](%[[VAL_SHAPE]]) typeparams %[[VAL_LEN]] {adapt.valuebyref} : (!hlfir.expr<?x!fir.char<1,?>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1)
! CHECK:    %[[VAL_BCAST:.*]] = fir.convert %[[VAL_ASSOC]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:    %[[VAL_EBC:.*]] = fir.emboxchar %[[VAL_BCAST]], %[[VAL_LEN]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:    fir.result %[[VAL_EBC]], %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.boxchar<1>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1
! CHECK:  } else {
! CHECK:    %[[ABSENT_0:.*]] = fir.absent !fir.boxchar<1>
! CHECK:    %[[ABSENT_1:.*]] = fir.absent !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:    %[[FALSE:.*]] = arith.constant false
! CHECK:    fir.result %[[ABSENT_0]], %[[ABSENT_1]], %[[FALSE]] : !fir.boxchar<1>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPdyn_char_array(%[[VAL_4]]#0, %[[VAL_N]]#0) {{.*}}: (!fir.boxchar<1>, !fir.ref<i64>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_4]]#1, %[[VAL_4]]#2 : !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1
end subroutine
end
