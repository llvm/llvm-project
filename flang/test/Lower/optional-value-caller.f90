! Test lowering of OPTIONAL VALUE dummy argument on caller side.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

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
! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_1:.*]] = arith.constant 42 : i32
! CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:  fir.call @_QPscalar(%[[VAL_0]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar(i)
  integer, optional :: i
  call scalar(i)
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.ref<i32>) {
! CHECK:    %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:    fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_1]] : !fir.ref<i32>
! CHECK:  } else {
! CHECK:    %[[VAL_5:.*]] = fir.absent !fir.ref<i32>
! CHECK:    fir.result %[[VAL_5]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_6:.*]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar2(i)
  integer, optional, value :: i
  call scalar(i)
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.ref<i32>) {
! CHECK:    %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:    fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_1]] : !fir.ref<i32>
! CHECK:  } else {
! CHECK:    %[[VAL_5:.*]] = fir.absent !fir.ref<i32>
! CHECK:    fir.result %[[VAL_5]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_6:.*]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar3(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i", fir.optional}) {
subroutine test_scalar3(i)
  integer, optional :: i
  ! i must be present when it appears in "()"
  call scalar((i))
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = fir.no_reassoc %[[VAL_2]] : i32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:  fir.call @_QPscalar(%[[VAL_1]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "i"}) {
subroutine test_scalar_ptr(i)
  integer, pointer :: i
  call scalar(i)
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<i32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_6]] -> (!fir.ref<i32>) {
! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
! CHECK:    fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_1]] : !fir.ref<i32>
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = fir.absent !fir.ref<i32>
! CHECK:    fir.result %[[VAL_11]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_12:.*]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_scalar_simple_var(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
subroutine test_scalar_simple_var(i)
  integer :: i
  call scalar(i)
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:  fir.call @_QPscalar(%[[VAL_1]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine


! CHECK-LABEL: func @_QMtestPtest_scalar_alloc(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "i"}) {
subroutine test_scalar_alloc(i)
  integer, allocatable :: i
  call scalar(i)
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<i32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_6]] -> (!fir.ref<i32>) {
! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.heap<i32>
! CHECK:    fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_1]] : !fir.ref<i32>
! CHECK:  } else {
! CHECK:    %[[VAL_11:.*]] = fir.absent !fir.ref<i32>
! CHECK:    fir.result %[[VAL_11]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_12:.*]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_ptr_2() {
subroutine test_ptr_2()
  call scalar(returns_ptr())
! CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = ".result"}
! CHECK:  %[[VAL_2:.*]] = fir.call @_QPreturns_ptr() {{.*}}: () -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.save_result %[[VAL_2]] to %[[VAL_1]] : !fir.box<!fir.ptr<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<i32>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_10:.*]] = fir.if %[[VAL_7]] -> (!fir.ref<i32>) {
! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ptr<i32>
! CHECK:    fir.store %[[VAL_11]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_0]] : !fir.ref<i32>
! CHECK:  } else {
! CHECK:    %[[VAL_12:.*]] = fir.absent !fir.ref<i32>
! CHECK:    fir.result %[[VAL_12]] : !fir.ref<i32>
! CHECK:  }
! CHECK:  fir.call @_QPscalar(%[[VAL_13:.*]]) {{.*}}: (!fir.ref<i32>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "i", fir.optional}) {
subroutine test_array(i)
  integer, optional :: i(100)
  call array(i)
! CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<!fir.array<100xi32>>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.heap<!fir.array<100xi32>>) {
! CHECK:    %[[VAL_4:.*]] = fir.allocmem !fir.array<100xi32>, %[[VAL_1]] {uniq_name = ".copy"}
! CHECK:    %[[VAL_5:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.heap<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:    %[[VAL_7:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_8:.*]] = fir.array_load %[[VAL_0]](%[[VAL_7]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:    %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_11:.*]] = arith.subi %[[VAL_1]], %[[VAL_9]] : index
! CHECK:    %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_6]]) -> (!fir.array<100xi32>) {
! CHECK:      %[[VAL_15:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_13]] : (!fir.array<100xi32>, index) -> i32
! CHECK:      %[[VAL_16:.*]] = fir.array_update %[[VAL_14]], %[[VAL_15]], %[[VAL_13]] : (!fir.array<100xi32>, i32, index) -> !fir.array<100xi32>
! CHECK:      fir.result %[[VAL_16]] : !fir.array<100xi32>
! CHECK:    }
! CHECK:    fir.array_merge_store %[[VAL_6]], %[[VAL_17:.*]] to %[[VAL_4]] : !fir.array<100xi32>, !fir.array<100xi32>, !fir.heap<!fir.array<100xi32>>
! CHECK:    fir.result %[[VAL_4]] : !fir.heap<!fir.array<100xi32>>
! CHECK:  } else {
! CHECK:    %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<100xi32>>
! CHECK:    fir.result %[[VAL_18]] : !fir.heap<!fir.array<100xi32>>
! CHECK:  }
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_20:.*]] : (!fir.heap<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:  fir.call @_QParray(%[[VAL_19]]) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:  fir.if %[[VAL_2]] {
! CHECK:    fir.freemem %[[VAL_20]] : !fir.heap<!fir.array<100xi32>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
subroutine test_array2(i, n)
  integer(8) :: n
  integer, optional, value :: i(n)
  call array(i)
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:    %[[VAL_9:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_6]] {uniq_name = ".copy"}
! CHECK:    %[[VAL_17:.*]] = fir.do_loop
! CHECK:    }
! CHECK:    fir.array_merge_store %{{.*}}, %[[VAL_17]] to %[[VAL_9]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
! CHECK:    fir.result %[[VAL_9]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  } else {
! CHECK:    %[[VAL_23:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:    fir.result %[[VAL_23]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:  fir.call @_QParray(%[[VAL_24]]) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:  fir.if %[[VAL_7]] {
! CHECK:    fir.freemem %[[VAL_8]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMtestPtest_dyn_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
subroutine test_dyn_array(i, n)
  integer(8) :: n
  integer, optional :: i(n)
  call dyn_array(i, n)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_3]], %[[VAL_4]] : index
! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:    %[[VAL_9:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {uniq_name = ".copy"}
! CHECK:    %[[VAL_17:.*]] = fir.do_loop
! CHECK:    }
! CHECK:    fir.array_merge_store %{{.*}}, %[[VAL_17]] to %[[VAL_9]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
! CHECK:    fir.result %[[VAL_9]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  } else {
! CHECK:    %[[VAL_23:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:    fir.result %[[VAL_23]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:  fir.call @_QPdyn_array(%[[VAL_24]], %[[VAL_1]]) {{.*}}: (!fir.ref<!fir.array<?xi32>>, !fir.ref<i64>) -> ()
! CHECK:  fir.if %[[VAL_7]] {
! CHECK:    fir.freemem %[[VAL_8]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMtestPtest_dyn_array_from_assumed(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
subroutine test_dyn_array_from_assumed(i, n)
  integer(8) :: n
  integer, optional :: i(:)
  call dyn_array(i, n)
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> i1
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:  %[[VAL_7:.*]] = arith.select %[[VAL_2]], %[[VAL_0]], %[[VAL_6]] : !fir.box<!fir.array<?xi32>>
! CHECK:  %[[box_none:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK:  %[[VAL_8:.*]] = fir.if %[[VAL_2]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:    %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_9]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:    %[[VAL_11:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]]#1 {uniq_name = ".copy"}
! CHECK:    fir.call @_FortranAAssign
! CHECK:    fir.result %[[VAL_11]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  } else {
! CHECK:    %[[VAL_24:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:    fir.result %[[VAL_24]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
! CHECK:  %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false{{.*}} : i1
! CHECK:  %[[and:.*]] = arith.andi %[[VAL_2]], %[[not_contiguous]] : i1
! CHECK:  %[[VAL_25:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:  fir.call @_QPdyn_array(%[[VAL_25]], %[[VAL_1]]) {{.*}}: (!fir.ref<!fir.array<?xi32>>, !fir.ref<i64>) -> ()
! CHECK:  fir.if %[[and]] {
! CHECK-NOT: fir.call @_FortranACopyOutAssign
! CHECK:    fir.freemem %[[VAL_8]] : !fir.heap<!fir.array<?xi32>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMtestPtest_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "i"}) {
subroutine test_array_ptr(i)
  integer, pointer :: i(:)
  call array(i)
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:         %[[VAL_9:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_8]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_10:.*]] = fir.if %[[VAL_6]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:           %[[VAL_11:.*]] = fir.if %[[VAL_9]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:             %[[VAL_12:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.result %[[VAL_12]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           } else {
! CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_13]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_15:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_14]]#1 {uniq_name = ".copy"}
! CHECK:             %[[VAL_16:.*]] = fir.shape %[[VAL_14]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:             fir.store %[[VAL_17]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:             %[[VAL_18:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:             %[[VAL_19:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_23:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:             fir.result %[[VAL_15]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_24:.*]] : !fir.heap<!fir.array<?xi32>>
! CHECK:         } else {
! CHECK:           %[[VAL_25:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.result %[[VAL_25]] : !fir.heap<!fir.array<?xi32>>
! CHECK:         }
! CHECK:         %[[VAL_26:.*]] = arith.constant false
! CHECK:         %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_26]] : i1
! CHECK:         %[[VAL_28:.*]] = arith.andi %[[VAL_6]], %[[VAL_27]] : i1
! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_30:.*]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:         fir.call @_QParray(%[[VAL_29]]) fastmath<contract> : (!fir.ref<!fir.array<100xi32>>) -> ()
! CHECK:         fir.if %[[VAL_28]] {
! CHECK:           fir.freemem %[[VAL_30]] : !fir.heap<!fir.array<?xi32>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c", fir.optional}) {
subroutine test_char(c)
  character(*), optional :: c
  call dyn_char(c)
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> i1
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]] = arith.select %[[VAL_2]], %[[VAL_1]]#1, %[[VAL_3]] : index
! CHECK:  %[[VAL_5:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_4]] : index) {adapt.valuebyref}
! CHECK:  %[[VAL_6:.*]] = fir.if %[[VAL_2]] -> (!fir.ref<!fir.char<1,?>>) {
! CHECK:    %[[VAL_13:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    %[[VAL_14:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    fir.call @llvm.memmove.p0.p0.i64(%[[VAL_13]], %[[VAL_14]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:    fir.result %[[VAL_5]] : !fir.ref<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_24:.*]] = fir.absent !fir.ref<!fir.char<1,?>>
! CHECK:    fir.result %[[VAL_24]] : !fir.ref<!fir.char<1,?>>
! CHECK:  }
! CHECK:  %[[VAL_25:.*]] = fir.emboxchar %[[VAL_6]], %[[VAL_4]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPdyn_char(%[[VAL_25]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>> {fir.bindc_name = "c"}) {
subroutine test_char_ptr(c)
  character(:), pointer :: c
  call dyn_char(c)
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:  %[[VAL_7:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_10:.*]] = arith.select %[[VAL_5]], %[[VAL_7]], %[[VAL_9]] : index
! CHECK:  %[[VAL_11:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_10]] : index) {adapt.valuebyref}
! CHECK:  %[[VAL_12:.*]] = fir.if %[[VAL_5]] -> (!fir.ref<!fir.char<1,?>>) {
! CHECK:    %[[VAL_19:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    %[[VAL_20:.*]] = fir.convert %[[VAL_8]] : (!fir.ptr<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    fir.call @llvm.memmove.p0.p0.i64(%[[VAL_19]], %[[VAL_20]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:    fir.result %[[VAL_11]] : !fir.ref<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_30:.*]] = fir.absent !fir.ref<!fir.char<1,?>>
! CHECK:    fir.result %[[VAL_30]] : !fir.ref<!fir.char<1,?>>
! CHECK:  }
! CHECK:  %[[VAL_31:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_10]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPdyn_char(%[[VAL_31]]) {{.*}}: (!fir.boxchar<1>) -> ()
end subroutine

! CHECK-LABEL: func @_QMtestPtest_char_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "c", fir.optional}) {
subroutine test_char_array(c)
  integer(8) :: n
  character(*), optional :: c(:)
  call dyn_char_array(c, n)
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_2:.*]] = fir.alloca i64 {bindc_name = "n", uniq_name = "_QMtestFtest_char_arrayEn"}
! CHECK:         %[[VAL_3:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
! CHECK:         %[[VAL_4:.*]] = fir.zero_bits !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) typeparams %[[VAL_7]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_9:.*]] = arith.select %[[VAL_3]], %[[VAL_0]], %[[VAL_8]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_10]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:         %[[VAL_12:.*]] = fir.if %[[VAL_3]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:           %[[VAL_13:.*]] = fir.if %[[VAL_11]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:             %[[VAL_14:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:             fir.result %[[VAL_14]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           } else {
! CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_15]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_17:.*]] = fir.box_elesize %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:             %[[VAL_18:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_17]] : index), %[[VAL_16]]#1 {uniq_name = ".copy"}
! CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_16]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_20:.*]] = fir.embox %[[VAL_18]](%[[VAL_19]]) typeparams %[[VAL_17]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:             fir.store %[[VAL_20]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>
! CHECK:             %[[VAL_21:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:             %[[VAL_22:.*]] = arith.constant {{.*}} : i32
! CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_26:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_22]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:             fir.result %[[VAL_18]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_27:.*]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         } else {
! CHECK:           %[[VAL_28:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           fir.result %[[VAL_28]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         %[[VAL_29:.*]] = fir.box_elesize %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_30:.*]] = arith.constant false
! CHECK:         %[[VAL_31:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_30]] : i1
! CHECK:         %[[VAL_32:.*]] = arith.andi %[[VAL_3]], %[[VAL_31]] : i1
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_34:.*]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_35:.*]] = fir.emboxchar %[[VAL_33]], %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         fir.call @_QPdyn_char_array(%[[VAL_35]], %[[VAL_2]]) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i64>) -> ()
! CHECK:         fir.if %[[VAL_32]] {
! CHECK:           fir.freemem %[[VAL_34]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine
end
