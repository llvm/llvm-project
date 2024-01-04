! Test allocatable assignments
! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

module alloc_assign
  type t
    integer :: i
  end type
contains

! -----------------------------------------------------------------------------
!            Test simple scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_simple_scalar(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>{{.*}}) {
subroutine test_simple_scalar(x)
  real, allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<f32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:2 = fir.if %[[VAL_6]] -> (i1, !fir.heap<f32>) {
! CHECK:    %[[VAL_8:.*]] = arith.constant false
! CHECK:    %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.heap<f32>) {
! CHECK:      %[[VAL_10:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_10]] : !fir.heap<f32>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_3]] : !fir.heap<f32>
! CHECK:    }
! CHECK:    fir.result %[[VAL_8]], %[[VAL_11:.*]] : i1, !fir.heap<f32>
! CHECK:  } else {
! CHECK:    %[[VAL_12:.*]] = arith.constant true
! CHECK:    %[[VAL_13:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_12]], %[[VAL_13]] : i1, !fir.heap<f32>
! CHECK:  }
! CHECK:  fir.store %[[VAL_1]] to %[[VAL_14:.*]]#1 : !fir.heap<f32>
! CHECK:  fir.if %[[VAL_14]]#0 {
! CHECK:    fir.if %[[VAL_6]] {
! CHECK:      fir.freemem %[[VAL_3]]
! CHECK:    }
! CHECK:    %[[VAL_15:.*]] = fir.embox %[[VAL_14]]#1 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:    fir.store %[[VAL_15]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  }
  x = 42.
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_simple_local_scalar() {
subroutine test_simple_local_scalar()
  real, allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.heap<f32> {uniq_name = "_QMalloc_assignFtest_simple_local_scalarEx.addr"}
! CHECK:  %[[VAL_2:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<f32>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:  %[[VAL_8:.*]]:2 = fir.if %[[VAL_7]] -> (i1, !fir.heap<f32>) {
! CHECK:    %[[VAL_9:.*]] = arith.constant false
! CHECK:    %[[VAL_10:.*]] = fir.if %[[VAL_9]] -> (!fir.heap<f32>) {
! CHECK:      %[[VAL_11:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_11]] : !fir.heap<f32>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_4]] : !fir.heap<f32>
! CHECK:    }
! CHECK:    fir.result %[[VAL_9]], %[[VAL_12:.*]] : i1, !fir.heap<f32>
! CHECK:  } else {
! CHECK:    %[[VAL_13:.*]] = arith.constant true
! CHECK:    %[[VAL_14:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_13]], %[[VAL_14]] : i1, !fir.heap<f32>
! CHECK:  }
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_15:.*]]#1 : !fir.heap<f32>
! CHECK:  fir.if %[[VAL_15]]#0 {
! CHECK:    fir.if %[[VAL_7]] {
! CHECK:      fir.freemem %[[VAL_4]]
! CHECK:    }
! CHECK:    fir.store %[[VAL_15]]#1 to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  }
  x = 42.
end subroutine

! -----------------------------------------------------------------------------
!            Test character scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_deferred_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}}) {
subroutine test_deferred_char_scalar(x)
  character(:), allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:  %[[VAL_8:.*]]:2 = fir.if %[[VAL_7]] -> (i1, !fir.heap<!fir.char<1,?>>) {
! CHECK:    %[[VAL_9:.*]] = arith.constant false
! CHECK:    %[[VAL_10:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:    %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_2]] : index
! CHECK:    %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_11]], %[[VAL_9]] : i1
! CHECK:    %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (!fir.heap<!fir.char<1,?>>) {
! CHECK:      %[[VAL_14:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_2]] : index) {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_14]] : !fir.heap<!fir.char<1,?>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_4]] : !fir.heap<!fir.char<1,?>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_12]], %[[VAL_15:.*]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_16:.*]] = arith.constant true
! CHECK:    %[[VAL_17:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_2]] : index) {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_16]], %[[VAL_17]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_8]]#0 {
! CHECK:    fir.if %[[VAL_7]] {
! CHECK:      fir.freemem %[[VAL_4]]
! CHECK:    }
! CHECK:    %[[VAL_36:.*]] = fir.embox %[[VAL_8]]#1 typeparams %[[VAL_2]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:    fir.store %[[VAL_36]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>{{.*}}) {
subroutine test_cst_char_scalar(x)
  character(10), allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.heap<!fir.char<1,10>>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]]:2 = fir.if %[[VAL_8]] -> (i1, !fir.heap<!fir.char<1,10>>) {
! CHECK:    %[[VAL_10:.*]] = arith.constant false
! CHECK:    %[[VAL_11:.*]] = fir.if %[[VAL_10]] -> (!fir.heap<!fir.char<1,10>>) {
! CHECK:      %[[VAL_12:.*]] = fir.allocmem !fir.char<1,10> {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_12]] : !fir.heap<!fir.char<1,10>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_5]] : !fir.heap<!fir.char<1,10>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_10]], %[[VAL_13:.*]] : i1, !fir.heap<!fir.char<1,10>>
! CHECK:  } else {
! CHECK:    %[[VAL_14:.*]] = arith.constant true
! CHECK:    %[[VAL_15:.*]] = fir.allocmem !fir.char<1,10> {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_14]], %[[VAL_15]] : i1, !fir.heap<!fir.char<1,10>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_9]]#0 {
! CHECK:    fir.if %[[VAL_8]] {
! CHECK:      fir.freemem %[[VAL_5]]
! CHECK:    }
! CHECK:    %[[VAL_34:.*]] = fir.embox %[[VAL_9]]#1 : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:    fir.store %[[VAL_34]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_dyn_char_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x
! CHECK:  %[[VAL_2A:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2B:.*]] = arith.cmpi sgt, %[[VAL_2A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_2:.*]] = arith.select %[[VAL_2B]], %[[VAL_2A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_3:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_8:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:  %[[VAL_10:.*]]:2 = fir.if %[[VAL_9]] -> (i1, !fir.heap<!fir.char<1,?>>) {
! CHECK:    %[[VAL_11:.*]] = arith.constant false
! CHECK:    %[[VAL_12:.*]] = fir.if %[[VAL_11]] -> (!fir.heap<!fir.char<1,?>>) {
! CHECK:      %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:      %[[VAL_14:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_13]] : index) {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_14]] : !fir.heap<!fir.char<1,?>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_6]] : !fir.heap<!fir.char<1,?>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_11]], %[[VAL_15:.*]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_16:.*]] = arith.constant true
! CHECK:    %[[VAL_17:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:    %[[VAL_18:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_17]] : index) {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_16]], %[[VAL_18]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_10]]#0 {
! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:    fir.if %[[VAL_9]] {
! CHECK:      fir.freemem %[[VAL_6]]
! CHECK:    }
! CHECK:    %[[VAL_40:.*]] = fir.embox %[[VAL_10]]#1 typeparams %[[VAL_39]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:    fir.store %[[VAL_40]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_derived_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>{{.*}}) {
subroutine test_derived_scalar(x, s)
  type(t), allocatable  :: x
  type(t) :: s
  x = s
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>) -> !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:2 = fir.if %[[VAL_6]] -> (i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) {
! CHECK:    %[[VAL_8:.*]] = arith.constant false
! CHECK:    %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) {
! CHECK:      %[[VAL_10:.*]] = fir.allocmem !fir.type<_QMalloc_assignTt{i:i32}> {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_10]] : !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_3]] : !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_8]], %[[VAL_11:.*]] : i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  } else {
! CHECK:    %[[VAL_12:.*]] = arith.constant true
! CHECK:    %[[VAL_13:.*]] = fir.allocmem !fir.type<_QMalloc_assignTt{i:i32}> {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_12]], %[[VAL_13]] : i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  }
! CHECK:  %[[VAL_14:.*]] = fir.field_index i, !fir.type<_QMalloc_assignTt{i:i32}>
! CHECK:  %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:  %[[VAL_14b:.*]] = fir.field_index i, !fir.type<_QMalloc_assignTt{i:i32}>
! CHECK:  %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_7]]#1, %[[VAL_14b]] : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:  %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:  fir.store %[[VAL_17]] to %[[VAL_16]] : !fir.ref<i32
! CHECK:  fir.if %[[VAL_7]]#0 {
! CHECK:    fir.if %[[VAL_6]] {
! CHECK:      fir.freemem %[[VAL_3]]
! CHECK:    }
! CHECK:    %[[VAL_19:.*]] = fir.embox %[[VAL_7]]#1 : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) -> !fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>
! CHECK:    fir.store %[[VAL_19]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
! CHECK:  }
end subroutine

! -----------------------------------------------------------------------------
!            Test numeric/logical array RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_from_cst_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}}) {
subroutine test_from_cst_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(2, 3)
! CHECK:         %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_1]](%[[VAL_6]]) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.array<2x3xf32>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:         %[[VAL_13:.*]]:2 = fir.if %[[VAL_12]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:           %[[VAL_14:.*]] = arith.constant false
! CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_17]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_16]]#1, %[[VAL_4]] : index
! CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_19]], %[[VAL_14]] : i1
! CHECK:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_18]]#1, %[[VAL_5]] : index
! CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_21]], %[[VAL_20]] : i1
! CHECK:           %[[VAL_23:.*]] = fir.if %[[VAL_22]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_24:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_4]], %[[VAL_5]] {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_25:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_26:.*]] = fir.array_load %[[VAL_24]](%[[VAL_25]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_29:.*]] = arith.subi %[[VAL_4]], %[[VAL_27]] : index
! CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_5]], %[[VAL_27]] : index
! CHECK:             %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_28]] to %[[VAL_30]] step %[[VAL_27]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_26]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_34:.*]] = fir.do_loop %[[VAL_35:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_36:.*]] = %[[VAL_33]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_37:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_35]], %[[VAL_32]] : (!fir.array<2x3xf32>, index, index) -> f32
! CHECK:                 %[[VAL_38:.*]] = fir.array_update %[[VAL_36]], %[[VAL_37]], %[[VAL_35]], %[[VAL_32]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_38]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_39:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_26]], %[[VAL_40:.*]] to %[[VAL_24]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_24]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_41:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_42:.*]] = fir.array_load %[[VAL_9]](%[[VAL_41]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_43:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_44:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_4]], %[[VAL_43]] : index
! CHECK:             %[[VAL_46:.*]] = arith.subi %[[VAL_5]], %[[VAL_43]] : index
! CHECK:             %[[VAL_47:.*]] = fir.do_loop %[[VAL_48:.*]] = %[[VAL_44]] to %[[VAL_46]] step %[[VAL_43]] unordered iter_args(%[[VAL_49:.*]] = %[[VAL_42]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_50:.*]] = fir.do_loop %[[VAL_51:.*]] = %[[VAL_44]] to %[[VAL_45]] step %[[VAL_43]] unordered iter_args(%[[VAL_52:.*]] = %[[VAL_49]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_53:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_51]], %[[VAL_48]] : (!fir.array<2x3xf32>, index, index) -> f32
! CHECK:                 %[[VAL_54:.*]] = fir.array_update %[[VAL_52]], %[[VAL_53]], %[[VAL_51]], %[[VAL_48]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_54]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_55:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_42]], %[[VAL_56:.*]] to %[[VAL_9]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_9]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_22]], %[[VAL_57:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_58:.*]] = arith.constant true
! CHECK:           %[[VAL_59:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_4]], %[[VAL_5]] {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_60:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_61:.*]] = fir.array_load %[[VAL_59]](%[[VAL_60]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:           %[[VAL_62:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_63:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_64:.*]] = arith.subi %[[VAL_4]], %[[VAL_62]] : index
! CHECK:           %[[VAL_65:.*]] = arith.subi %[[VAL_5]], %[[VAL_62]] : index
! CHECK:           %[[VAL_66:.*]] = fir.do_loop %[[VAL_67:.*]] = %[[VAL_63]] to %[[VAL_65]] step %[[VAL_62]] unordered iter_args(%[[VAL_68:.*]] = %[[VAL_61]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_69:.*]] = fir.do_loop %[[VAL_70:.*]] = %[[VAL_63]] to %[[VAL_64]] step %[[VAL_62]] unordered iter_args(%[[VAL_71:.*]] = %[[VAL_68]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_72:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_70]], %[[VAL_67]] : (!fir.array<2x3xf32>, index, index) -> f32
! CHECK:               %[[VAL_73:.*]] = fir.array_update %[[VAL_71]], %[[VAL_72]], %[[VAL_70]], %[[VAL_67]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_73]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_74:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_61]], %[[VAL_75:.*]] to %[[VAL_59]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.result %[[VAL_58]], %[[VAL_59]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_76:.*]]#0 {
! CHECK:           fir.if %[[VAL_12]] {
! CHECK:             fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_77:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_78:.*]] = fir.embox %[[VAL_76]]#1(%[[VAL_77]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_78]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_from_dyn_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine test_from_dyn_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(:, :)
  x = y
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:         %[[VAL_12:.*]]:2 = fir.if %[[VAL_11]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:           %[[VAL_13:.*]] = arith.constant false
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_15]]#1, %[[VAL_3]]#1 : index
! CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_18]], %[[VAL_13]] : i1
! CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_17]]#1, %[[VAL_5]]#1 : index
! CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_20]], %[[VAL_19]] : i1
! CHECK:           %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_23:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_3]]#1, %[[VAL_5]]#1 {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_24:.*]] = fir.shape %[[VAL_3]]#1, %[[VAL_5]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_25:.*]] = fir.array_load %[[VAL_23]](%[[VAL_24]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_28:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_26]] : index
! CHECK:             %[[VAL_29:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_26]] : index
! CHECK:             %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_27]] to %[[VAL_29]] step %[[VAL_26]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_25]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_26]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_32]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_36:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_34]], %[[VAL_31]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_37:.*]] = fir.array_update %[[VAL_35]], %[[VAL_36]], %[[VAL_34]], %[[VAL_31]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_37]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_38:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_25]], %[[VAL_39:.*]] to %[[VAL_23]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_23]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_40:.*]] = fir.shape %[[VAL_3]]#1, %[[VAL_5]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_41:.*]] = fir.array_load %[[VAL_8]](%[[VAL_40]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_43:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_42]] : index
! CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_42]] : index
! CHECK:             %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %[[VAL_43]] to %[[VAL_45]] step %[[VAL_42]] unordered iter_args(%[[VAL_48:.*]] = %[[VAL_41]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_49:.*]] = fir.do_loop %[[VAL_50:.*]] = %[[VAL_43]] to %[[VAL_44]] step %[[VAL_42]] unordered iter_args(%[[VAL_51:.*]] = %[[VAL_48]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_52:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_50]], %[[VAL_47]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_53:.*]] = fir.array_update %[[VAL_51]], %[[VAL_52]], %[[VAL_50]], %[[VAL_47]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_53]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_54:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_41]], %[[VAL_55:.*]] to %[[VAL_8]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_8]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_21]], %[[VAL_56:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_57:.*]] = arith.constant true
! CHECK:           %[[VAL_58:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_3]]#1, %[[VAL_5]]#1 {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_59:.*]] = fir.shape %[[VAL_3]]#1, %[[VAL_5]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_60:.*]] = fir.array_load %[[VAL_58]](%[[VAL_59]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:           %[[VAL_61:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_62:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_63:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_61]] : index
! CHECK:           %[[VAL_64:.*]] = arith.subi %[[VAL_5]]#1, %[[VAL_61]] : index
! CHECK:           %[[VAL_65:.*]] = fir.do_loop %[[VAL_66:.*]] = %[[VAL_62]] to %[[VAL_64]] step %[[VAL_61]] unordered iter_args(%[[VAL_67:.*]] = %[[VAL_60]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_68:.*]] = fir.do_loop %[[VAL_69:.*]] = %[[VAL_62]] to %[[VAL_63]] step %[[VAL_61]] unordered iter_args(%[[VAL_70:.*]] = %[[VAL_67]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_71:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_69]], %[[VAL_66]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_72:.*]] = fir.array_update %[[VAL_70]], %[[VAL_71]], %[[VAL_69]], %[[VAL_66]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_72]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_73:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_60]], %[[VAL_74:.*]] to %[[VAL_58]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.result %[[VAL_57]], %[[VAL_58]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_75:.*]]#0 {
! CHECK:           fir.if %[[VAL_11]] {
! CHECK:             fir.freemem %[[VAL_8]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_76:.*]] = fir.shape %[[VAL_3]]#1, %[[VAL_5]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_77:.*]] = fir.embox %[[VAL_75]]#1(%[[VAL_76]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_77]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_with_lbounds(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine test_with_lbounds(x, y)
  real, allocatable  :: x(:, :)
  real :: y(10:, 20:)
! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:         %[[VAL_4:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = fir.shift %[[VAL_3]], %[[VAL_5]] : (index, index) -> !fir.shift<2>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_7]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_9]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]](%[[VAL_6]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:         %[[VAL_17:.*]]:2 = fir.if %[[VAL_16]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:           %[[VAL_18:.*]] = arith.constant false
! CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_20]]#1, %[[VAL_8]]#1 : index
! CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_23]], %[[VAL_18]] : i1
! CHECK:           %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_22]]#1, %[[VAL_10]]#1 : index
! CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_25]], %[[VAL_24]] : i1
! CHECK:           %[[VAL_27:.*]] = fir.if %[[VAL_26]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_28:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_8]]#1, %[[VAL_10]]#1 {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_29:.*]] = fir.shape %[[VAL_8]]#1, %[[VAL_10]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_30:.*]] = fir.array_load %[[VAL_28]](%[[VAL_29]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_32:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_33:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_31]] : index
! CHECK:             %[[VAL_34:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_31]] : index
! CHECK:             %[[VAL_35:.*]] = fir.do_loop %[[VAL_36:.*]] = %[[VAL_32]] to %[[VAL_34]] step %[[VAL_31]] unordered iter_args(%[[VAL_37:.*]] = %[[VAL_30]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_32]] to %[[VAL_33]] step %[[VAL_31]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_37]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_41:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_39]], %[[VAL_36]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_42:.*]] = fir.array_update %[[VAL_40]], %[[VAL_41]], %[[VAL_39]], %[[VAL_36]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_42]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_43:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_30]], %[[VAL_44:.*]] to %[[VAL_28]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_28]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_45:.*]] = fir.shape %[[VAL_8]]#1, %[[VAL_10]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_46:.*]] = fir.array_load %[[VAL_13]](%[[VAL_45]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_47:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_48:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_49:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_47]] : index
! CHECK:             %[[VAL_50:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_47]] : index
! CHECK:             %[[VAL_51:.*]] = fir.do_loop %[[VAL_52:.*]] = %[[VAL_48]] to %[[VAL_50]] step %[[VAL_47]] unordered iter_args(%[[VAL_53:.*]] = %[[VAL_46]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_54:.*]] = fir.do_loop %[[VAL_55:.*]] = %[[VAL_48]] to %[[VAL_49]] step %[[VAL_47]] unordered iter_args(%[[VAL_56:.*]] = %[[VAL_53]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_57:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_55]], %[[VAL_52]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_58:.*]] = fir.array_update %[[VAL_56]], %[[VAL_57]], %[[VAL_55]], %[[VAL_52]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_58]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_59:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_46]], %[[VAL_60:.*]] to %[[VAL_13]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_13]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_26]], %[[VAL_61:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_62:.*]] = arith.constant true
! CHECK:           %[[VAL_63:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_8]]#1, %[[VAL_10]]#1 {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_64:.*]] = fir.shape %[[VAL_8]]#1, %[[VAL_10]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_65:.*]] = fir.array_load %[[VAL_63]](%[[VAL_64]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:           %[[VAL_66:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_67:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_68:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_66]] : index
! CHECK:           %[[VAL_69:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_66]] : index
! CHECK:           %[[VAL_70:.*]] = fir.do_loop %[[VAL_71:.*]] = %[[VAL_67]] to %[[VAL_69]] step %[[VAL_66]] unordered iter_args(%[[VAL_72:.*]] = %[[VAL_65]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_73:.*]] = fir.do_loop %[[VAL_74:.*]] = %[[VAL_67]] to %[[VAL_68]] step %[[VAL_66]] unordered iter_args(%[[VAL_75:.*]] = %[[VAL_72]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_76:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_74]], %[[VAL_71]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_77:.*]] = fir.array_update %[[VAL_75]], %[[VAL_76]], %[[VAL_74]], %[[VAL_71]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_77]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_78:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_65]], %[[VAL_79:.*]] to %[[VAL_63]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.result %[[VAL_62]], %[[VAL_63]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_80:.*]]#0 {
! CHECK:           fir.if %[[VAL_16]] {
! CHECK:             fir.freemem %[[VAL_13]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_81:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_8]]#1, %[[VAL_5]], %[[VAL_10]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_82:.*]] = fir.embox %[[VAL_80]]#1(%[[VAL_81]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_82]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_runtime_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}}) {
subroutine test_runtime_shape(x)
  real, allocatable  :: x(:, :)
  interface
   function return_pointer()
     real, pointer :: return_pointer(:, :)
   end function
  end interface
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>> {bindc_name = ".result"}
! CHECK:         %[[VAL_2:.*]] = fir.call @_QPreturn_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.save_result %[[VAL_2]] to %[[VAL_1]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_8:.*]] = fir.shift %[[VAL_5]]#0, %[[VAL_7]]#0 : (index, index) -> !fir.shift<2>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_9]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.shift<2>) -> !fir.array<?x?xf32>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:         %[[VAL_17:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_17]] : i64
! CHECK:         %[[VAL_19:.*]]:2 = fir.if %[[VAL_18]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:           %[[VAL_20:.*]] = arith.constant false
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_22]]#1, %[[VAL_10]]#1 : index
! CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_25]], %[[VAL_20]] : i1
! CHECK:           %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_24]]#1, %[[VAL_12]]#1 : index
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_27]], %[[VAL_26]] : i1
! CHECK:           %[[VAL_29:.*]] = fir.if %[[VAL_28]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:             %[[VAL_30:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_10]]#1, %[[VAL_12]]#1 {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_31:.*]] = fir.shape %[[VAL_10]]#1, %[[VAL_12]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_32:.*]] = fir.array_load %[[VAL_30]](%[[VAL_31]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_34:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_33]] : index
! CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_33]] : index
! CHECK:             %[[VAL_37:.*]] = fir.do_loop %[[VAL_38:.*]] = %[[VAL_34]] to %[[VAL_36]] step %[[VAL_33]] unordered iter_args(%[[VAL_39:.*]] = %[[VAL_32]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_40:.*]] = fir.do_loop %[[VAL_41:.*]] = %[[VAL_34]] to %[[VAL_35]] step %[[VAL_33]] unordered iter_args(%[[VAL_42:.*]] = %[[VAL_39]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_43:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_41]], %[[VAL_38]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_44:.*]] = fir.array_update %[[VAL_42]], %[[VAL_43]], %[[VAL_41]], %[[VAL_38]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_44]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_45:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_32]], %[[VAL_46:.*]] to %[[VAL_30]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_30]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_47:.*]] = fir.shape %[[VAL_10]]#1, %[[VAL_12]]#1 : (index, index) -> !fir.shape<2>
! CHECK:             %[[VAL_48:.*]] = fir.array_load %[[VAL_15]](%[[VAL_47]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:             %[[VAL_49:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_50:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_51:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_49]] : index
! CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_49]] : index
! CHECK:             %[[VAL_53:.*]] = fir.do_loop %[[VAL_54:.*]] = %[[VAL_50]] to %[[VAL_52]] step %[[VAL_49]] unordered iter_args(%[[VAL_55:.*]] = %[[VAL_48]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_56:.*]] = fir.do_loop %[[VAL_57:.*]] = %[[VAL_50]] to %[[VAL_51]] step %[[VAL_49]] unordered iter_args(%[[VAL_58:.*]] = %[[VAL_55]]) -> (!fir.array<?x?xf32>) {
! CHECK:                 %[[VAL_59:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_57]], %[[VAL_54]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:                 %[[VAL_60:.*]] = fir.array_update %[[VAL_58]], %[[VAL_59]], %[[VAL_57]], %[[VAL_54]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:                 fir.result %[[VAL_60]] : !fir.array<?x?xf32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_61:.*]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_48]], %[[VAL_62:.*]] to %[[VAL_15]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:             fir.result %[[VAL_15]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_28]], %[[VAL_63:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_64:.*]] = arith.constant true
! CHECK:           %[[VAL_65:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_10]]#1, %[[VAL_12]]#1 {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_66:.*]] = fir.shape %[[VAL_10]]#1, %[[VAL_12]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_67:.*]] = fir.array_load %[[VAL_65]](%[[VAL_66]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! CHECK:           %[[VAL_68:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_69:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_70:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_68]] : index
! CHECK:           %[[VAL_71:.*]] = arith.subi %[[VAL_12]]#1, %[[VAL_68]] : index
! CHECK:           %[[VAL_72:.*]] = fir.do_loop %[[VAL_73:.*]] = %[[VAL_69]] to %[[VAL_71]] step %[[VAL_68]] unordered iter_args(%[[VAL_74:.*]] = %[[VAL_67]]) -> (!fir.array<?x?xf32>) {
! CHECK:             %[[VAL_75:.*]] = fir.do_loop %[[VAL_76:.*]] = %[[VAL_69]] to %[[VAL_70]] step %[[VAL_68]] unordered iter_args(%[[VAL_77:.*]] = %[[VAL_74]]) -> (!fir.array<?x?xf32>) {
! CHECK:               %[[VAL_78:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_76]], %[[VAL_73]] : (!fir.array<?x?xf32>, index, index) -> f32
! CHECK:               %[[VAL_79:.*]] = fir.array_update %[[VAL_77]], %[[VAL_78]], %[[VAL_76]], %[[VAL_73]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
! CHECK:               fir.result %[[VAL_79]] : !fir.array<?x?xf32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_80:.*]] : !fir.array<?x?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_67]], %[[VAL_81:.*]] to %[[VAL_65]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.result %[[VAL_64]], %[[VAL_65]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_82:.*]]#0 {
! CHECK:           fir.if %[[VAL_18]] {
! CHECK:             fir.freemem %[[VAL_15]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_83:.*]] = fir.shape %[[VAL_10]]#1, %[[VAL_12]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_84:.*]] = fir.embox %[[VAL_82]]#1(%[[VAL_83]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_84]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
  x = return_pointer()
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_scalar_rhs(
subroutine test_scalar_rhs(x, y)
  real, allocatable  :: x(:)
  real :: y
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK: %[[error_msg_addr:.*]] = fir.address_of(@[[error_message:.*]]) : !fir.ref<!fir.char<1,76>>
  ! CHECK: %[[msg_addr_cast:.*]] = fir.convert %[[error_msg_addr]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
  ! CHECK: %{{.*}} = fir.call @_FortranAReportFatalUserError(%[[msg_addr_cast]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i32) -> none
  ! CHECK-NOT: allocmem
  ! CHECK: }
  x = y
end subroutine

! -----------------------------------------------------------------------------
!            Test character array RHS
! -----------------------------------------------------------------------------


! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char_rhs_scalar(x)
!  character(:), allocatable  :: x(:)
!  x = "Hello world!"
!end subroutine

! CHECK: func @_QMalloc_assignPtest_cst_char_rhs_scalar(
subroutine test_cst_char_rhs_scalar(x)
  character(10), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! CHECK: func @_QMalloc_assignPtest_dyn_char_rhs_scalar(
subroutine test_dyn_char_rhs_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char(x, c)
!  character(:), allocatable  :: x(:)
!  character(12) :: c(20)
!  x = "Hello world!"
!end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine test_cst_char(x, c)
  character(10), allocatable  :: x(:)
  character(12) :: c(20)
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,12>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 12 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_4]](%[[VAL_7]]) : (!fir.ref<!fir.array<20x!fir.char<1,12>>>, !fir.shape<1>) -> !fir.array<20x!fir.char<1,12>>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:         %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:         %[[VAL_14:.*]]:2 = fir.if %[[VAL_13]] -> (i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>) {
! CHECK:           %[[VAL_15:.*]] = arith.constant false
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_17]]#1, %[[VAL_6]] : index
! CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_18]], %[[VAL_15]] : i1
! CHECK:           %[[VAL_20:.*]] = fir.if %[[VAL_19]] -> (!fir.heap<!fir.array<?x!fir.char<1,10>>>) {
! CHECK:             %[[VAL_21:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_6]] {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_22:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_23:.*]] = fir.array_load %[[VAL_21]](%[[VAL_22]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_6]], %[[VAL_24]] : index
! CHECK:             %[[VAL_27:.*]] = fir.do_loop %[[VAL_28:.*]] = %[[VAL_25]] to %[[VAL_26]] step %[[VAL_24]] unordered iter_args(%[[VAL_29:.*]] = %[[VAL_23]]) -> (!fir.array<?x!fir.char<1,10>>) {
! CHECK:               %[[VAL_30:.*]] = fir.array_access %[[VAL_8]], %[[VAL_28]] : (!fir.array<20x!fir.char<1,12>>, index) -> !fir.ref<!fir.char<1,12>>
! CHECK:               %[[VAL_31:.*]] = fir.array_access %[[VAL_29]], %[[VAL_28]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:               %[[VAL_32:.*]] = arith.constant 10 : index
! CHECK:               %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_3]] : index
! CHECK:               %[[VAL_34:.*]] = arith.select %[[VAL_33]], %[[VAL_32]], %[[VAL_3]] : index
! CHECK:               %[[VAL_35:.*]] = arith.constant 1 : i64
! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
! CHECK:               %[[VAL_37:.*]] = arith.muli %[[VAL_35]], %[[VAL_36]] : i64
! CHECK:               %[[VAL_38:.*]] = arith.constant false
! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_40:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
! CHECK:               fir.call @llvm.memmove.p0.p0.i64(%[[VAL_39]], %[[VAL_40]], %[[VAL_37]], %[[VAL_38]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:               %[[VAL_41:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_42:.*]] = arith.subi %[[VAL_32]], %[[VAL_41]] : index
! CHECK:               %[[VAL_43:.*]] = arith.constant 32 : i8
! CHECK:               %[[VAL_44:.*]] = fir.undefined !fir.char<1>
! CHECK:               %[[VAL_45:.*]] = fir.insert_value %[[VAL_44]], %[[VAL_43]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:               %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:               fir.do_loop %[[VAL_47:.*]] = %[[VAL_34]] to %[[VAL_42]] step %[[VAL_46]] {
! CHECK:                 %[[VAL_48:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:                 %[[VAL_49:.*]] = fir.coordinate_of %[[VAL_48]], %[[VAL_47]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:                 fir.store %[[VAL_45]] to %[[VAL_49]] : !fir.ref<!fir.char<1>>
! CHECK:               }
! CHECK:               %[[VAL_50:.*]] = fir.array_amend %[[VAL_29]], %[[VAL_31]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:               fir.result %[[VAL_50]] : !fir.array<?x!fir.char<1,10>>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_23]], %[[VAL_51:.*]] to %[[VAL_21]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:             fir.result %[[VAL_21]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           } else {
! CHECK:             %[[VAL_52:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_53:.*]] = fir.array_load %[[VAL_10]](%[[VAL_52]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:             %[[VAL_54:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_55:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_56:.*]] = arith.subi %[[VAL_6]], %[[VAL_54]] : index
! CHECK:             %[[VAL_57:.*]] = fir.do_loop %[[VAL_58:.*]] = %[[VAL_55]] to %[[VAL_56]] step %[[VAL_54]] unordered iter_args(%[[VAL_59:.*]] = %[[VAL_53]]) -> (!fir.array<?x!fir.char<1,10>>) {
! CHECK:               %[[VAL_60:.*]] = fir.array_access %[[VAL_8]], %[[VAL_58]] : (!fir.array<20x!fir.char<1,12>>, index) -> !fir.ref<!fir.char<1,12>>
! CHECK:               %[[VAL_61:.*]] = fir.array_access %[[VAL_59]], %[[VAL_58]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:               %[[VAL_62:.*]] = arith.constant 10 : index
! CHECK:               %[[VAL_63:.*]] = arith.cmpi slt, %[[VAL_62]], %[[VAL_3]] : index
! CHECK:               %[[VAL_64:.*]] = arith.select %[[VAL_63]], %[[VAL_62]], %[[VAL_3]] : index
! CHECK:               %[[VAL_65:.*]] = arith.constant 1 : i64
! CHECK:               %[[VAL_66:.*]] = fir.convert %[[VAL_64]] : (index) -> i64
! CHECK:               %[[VAL_67:.*]] = arith.muli %[[VAL_65]], %[[VAL_66]] : i64
! CHECK:               %[[VAL_68:.*]] = arith.constant false
! CHECK:               %[[VAL_69:.*]] = fir.convert %[[VAL_61]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_70:.*]] = fir.convert %[[VAL_60]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
! CHECK:               fir.call @llvm.memmove.p0.p0.i64(%[[VAL_69]], %[[VAL_70]], %[[VAL_67]], %[[VAL_68]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:               %[[VAL_71:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_72:.*]] = arith.subi %[[VAL_62]], %[[VAL_71]] : index
! CHECK:               %[[VAL_73:.*]] = arith.constant 32 : i8
! CHECK:               %[[VAL_74:.*]] = fir.undefined !fir.char<1>
! CHECK:               %[[VAL_75:.*]] = fir.insert_value %[[VAL_74]], %[[VAL_73]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:               %[[VAL_76:.*]] = arith.constant 1 : index
! CHECK:               fir.do_loop %[[VAL_77:.*]] = %[[VAL_64]] to %[[VAL_72]] step %[[VAL_76]] {
! CHECK:                 %[[VAL_78:.*]] = fir.convert %[[VAL_61]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:                 %[[VAL_79:.*]] = fir.coordinate_of %[[VAL_78]], %[[VAL_77]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:                 fir.store %[[VAL_75]] to %[[VAL_79]] : !fir.ref<!fir.char<1>>
! CHECK:               }
! CHECK:               %[[VAL_80:.*]] = fir.array_amend %[[VAL_59]], %[[VAL_61]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:               fir.result %[[VAL_80]] : !fir.array<?x!fir.char<1,10>>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_53]], %[[VAL_81:.*]] to %[[VAL_10]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:             fir.result %[[VAL_10]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_19]], %[[VAL_82:.*]] : i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:         } else {
! CHECK:           %[[VAL_83:.*]] = arith.constant true
! CHECK:           %[[VAL_84:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_6]] {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_85:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_86:.*]] = fir.array_load %[[VAL_84]](%[[VAL_85]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:           %[[VAL_87:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_88:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_89:.*]] = arith.subi %[[VAL_6]], %[[VAL_87]] : index
! CHECK:           %[[VAL_90:.*]] = fir.do_loop %[[VAL_91:.*]] = %[[VAL_88]] to %[[VAL_89]] step %[[VAL_87]] unordered iter_args(%[[VAL_92:.*]] = %[[VAL_86]]) -> (!fir.array<?x!fir.char<1,10>>) {
! CHECK:             %[[VAL_93:.*]] = fir.array_access %[[VAL_8]], %[[VAL_91]] : (!fir.array<20x!fir.char<1,12>>, index) -> !fir.ref<!fir.char<1,12>>
! CHECK:             %[[VAL_94:.*]] = fir.array_access %[[VAL_92]], %[[VAL_91]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:             %[[VAL_95:.*]] = arith.constant 10 : index
! CHECK:             %[[VAL_96:.*]] = arith.cmpi slt, %[[VAL_95]], %[[VAL_3]] : index
! CHECK:             %[[VAL_97:.*]] = arith.select %[[VAL_96]], %[[VAL_95]], %[[VAL_3]] : index
! CHECK:             %[[VAL_98:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_99:.*]] = fir.convert %[[VAL_97]] : (index) -> i64
! CHECK:             %[[VAL_100:.*]] = arith.muli %[[VAL_98]], %[[VAL_99]] : i64
! CHECK:             %[[VAL_101:.*]] = arith.constant false
! CHECK:             %[[VAL_102:.*]] = fir.convert %[[VAL_94]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_103:.*]] = fir.convert %[[VAL_93]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
! CHECK:             fir.call @llvm.memmove.p0.p0.i64(%[[VAL_102]], %[[VAL_103]], %[[VAL_100]], %[[VAL_101]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:             %[[VAL_104:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_105:.*]] = arith.subi %[[VAL_95]], %[[VAL_104]] : index
! CHECK:             %[[VAL_106:.*]] = arith.constant 32 : i8
! CHECK:             %[[VAL_107:.*]] = fir.undefined !fir.char<1>
! CHECK:             %[[VAL_108:.*]] = fir.insert_value %[[VAL_107]], %[[VAL_106]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:             %[[VAL_109:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_110:.*]] = %[[VAL_97]] to %[[VAL_105]] step %[[VAL_109]] {
! CHECK:               %[[VAL_111:.*]] = fir.convert %[[VAL_94]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:               %[[VAL_112:.*]] = fir.coordinate_of %[[VAL_111]], %[[VAL_110]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_108]] to %[[VAL_112]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:             %[[VAL_113:.*]] = fir.array_amend %[[VAL_92]], %[[VAL_94]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:             fir.result %[[VAL_113]] : !fir.array<?x!fir.char<1,10>>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_86]], %[[VAL_114:.*]] to %[[VAL_84]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           fir.result %[[VAL_83]], %[[VAL_84]] : i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_115:.*]]#0 {
! CHECK:           fir.if %[[VAL_13]] {
! CHECK:             fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           }
! CHECK:           %[[VAL_116:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_117:.*]] = fir.embox %[[VAL_115]]#1(%[[VAL_116]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
! CHECK:           fir.store %[[VAL_117]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
  x = c
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine test_dyn_char(x, n, c)
  integer :: n
  character(n), allocatable  :: x(:)
  character(*) :: c(20)
! CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,?>>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:         %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : i32
! CHECK:         %[[VAL_10:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_4]](%[[VAL_11]]) typeparams %[[VAL_3]]#1 : (!fir.ref<!fir.array<20x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<20x!fir.char<1,?>>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> i64
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:         %[[VAL_18:.*]]:2 = fir.if %[[VAL_17]] -> (i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:           %[[VAL_19:.*]] = arith.constant false
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_21]]#1, %[[VAL_10]] : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_22]], %[[VAL_19]] : i1
! CHECK:           %[[VAL_24:.*]] = fir.if %[[VAL_23]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:             %[[VAL_26:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_25]] : index), %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_27:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_28:.*]] = fir.array_load %[[VAL_26]](%[[VAL_27]]) typeparams %[[VAL_9]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<?x!fir.char<1,?>>
! CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_10]], %[[VAL_29]] : index
! CHECK:             %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_28]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:               %[[VAL_35:.*]] = fir.array_access %[[VAL_12]], %[[VAL_33]] typeparams %[[VAL_3]]#1 : (!fir.array<20x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:               %[[VAL_36:.*]] = fir.array_access %[[VAL_34]], %[[VAL_33]] typeparams %[[VAL_9]] : (!fir.array<?x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:               %[[VAL_38:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_3]]#1 : index
! CHECK:               %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_37]], %[[VAL_3]]#1 : index
! CHECK:               %[[VAL_40:.*]] = arith.constant 1 : i64
! CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (index) -> i64
! CHECK:               %[[VAL_42:.*]] = arith.muli %[[VAL_40]], %[[VAL_41]] : i64
! CHECK:               %[[VAL_43:.*]] = arith.constant false
! CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:               fir.call @llvm.memmove.p0.p0.i64(%[[VAL_44]], %[[VAL_45]], %[[VAL_42]], %[[VAL_43]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:               %[[VAL_46:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_47:.*]] = arith.subi %[[VAL_9]], %[[VAL_46]] : i32
! CHECK:               %[[VAL_48:.*]] = arith.constant 32 : i8
! CHECK:               %[[VAL_49:.*]] = fir.undefined !fir.char<1>
! CHECK:               %[[VAL_50:.*]] = fir.insert_value %[[VAL_49]], %[[VAL_48]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:               %[[VAL_51:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_47]] : (i32) -> index
! CHECK:               fir.do_loop %[[VAL_53:.*]] = %[[VAL_39]] to %[[VAL_52]] step %[[VAL_51]] {
! CHECK:                 %[[VAL_54:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:                 %[[VAL_55:.*]] = fir.coordinate_of %[[VAL_54]], %[[VAL_53]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:                 fir.store %[[VAL_50]] to %[[VAL_55]] : !fir.ref<!fir.char<1>>
! CHECK:               }
! CHECK:               %[[VAL_56:.*]] = fir.array_amend %[[VAL_34]], %[[VAL_36]] : (!fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:               fir.result %[[VAL_56]] : !fir.array<?x!fir.char<1,?>>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_28]], %[[VAL_57:.*]] to %[[VAL_26]] typeparams %[[VAL_9]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, i32
! CHECK:             fir.result %[[VAL_26]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           } else {
! CHECK:             %[[VAL_58:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_59:.*]] = fir.array_load %[[VAL_14]](%[[VAL_58]]) typeparams %[[VAL_9]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<?x!fir.char<1,?>>
! CHECK:             %[[VAL_60:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_61:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_62:.*]] = arith.subi %[[VAL_10]], %[[VAL_60]] : index
! CHECK:             %[[VAL_63:.*]] = fir.do_loop %[[VAL_64:.*]] = %[[VAL_61]] to %[[VAL_62]] step %[[VAL_60]] unordered iter_args(%[[VAL_65:.*]] = %[[VAL_59]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:               %[[VAL_66:.*]] = fir.array_access %[[VAL_12]], %[[VAL_64]] typeparams %[[VAL_3]]#1 : (!fir.array<20x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:               %[[VAL_67:.*]] = fir.array_access %[[VAL_65]], %[[VAL_64]] typeparams %[[VAL_9]] : (!fir.array<?x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:               %[[VAL_68:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:               %[[VAL_69:.*]] = arith.cmpi slt, %[[VAL_68]], %[[VAL_3]]#1 : index
! CHECK:               %[[VAL_70:.*]] = arith.select %[[VAL_69]], %[[VAL_68]], %[[VAL_3]]#1 : index
! CHECK:               %[[VAL_71:.*]] = arith.constant 1 : i64
! CHECK:               %[[VAL_72:.*]] = fir.convert %[[VAL_70]] : (index) -> i64
! CHECK:               %[[VAL_73:.*]] = arith.muli %[[VAL_71]], %[[VAL_72]] : i64
! CHECK:               %[[VAL_74:.*]] = arith.constant false
! CHECK:               %[[VAL_75:.*]] = fir.convert %[[VAL_67]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_76:.*]] = fir.convert %[[VAL_66]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:               fir.call @llvm.memmove.p0.p0.i64(%[[VAL_75]], %[[VAL_76]], %[[VAL_73]], %[[VAL_74]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:               %[[VAL_77:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_78:.*]] = arith.subi %[[VAL_9]], %[[VAL_77]] : i32
! CHECK:               %[[VAL_79:.*]] = arith.constant 32 : i8
! CHECK:               %[[VAL_80:.*]] = fir.undefined !fir.char<1>
! CHECK:               %[[VAL_81:.*]] = fir.insert_value %[[VAL_80]], %[[VAL_79]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:               %[[VAL_82:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_83:.*]] = fir.convert %[[VAL_78]] : (i32) -> index
! CHECK:               fir.do_loop %[[VAL_84:.*]] = %[[VAL_70]] to %[[VAL_83]] step %[[VAL_82]] {
! CHECK:                 %[[VAL_85:.*]] = fir.convert %[[VAL_67]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:                 %[[VAL_86:.*]] = fir.coordinate_of %[[VAL_85]], %[[VAL_84]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:                 fir.store %[[VAL_81]] to %[[VAL_86]] : !fir.ref<!fir.char<1>>
! CHECK:               }
! CHECK:               %[[VAL_87:.*]] = fir.array_amend %[[VAL_65]], %[[VAL_67]] : (!fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:               fir.result %[[VAL_87]] : !fir.array<?x!fir.char<1,?>>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_59]], %[[VAL_88:.*]] to %[[VAL_14]] typeparams %[[VAL_9]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, i32
! CHECK:             fir.result %[[VAL_14]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_23]], %[[VAL_89:.*]] : i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         } else {
! CHECK:           %[[VAL_90:.*]] = arith.constant true
! CHECK:           %[[VAL_91:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           %[[VAL_92:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_91]] : index), %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_93:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_94:.*]] = fir.array_load %[[VAL_92]](%[[VAL_93]]) typeparams %[[VAL_9]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<?x!fir.char<1,?>>
! CHECK:           %[[VAL_95:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_96:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_97:.*]] = arith.subi %[[VAL_10]], %[[VAL_95]] : index
! CHECK:           %[[VAL_98:.*]] = fir.do_loop %[[VAL_99:.*]] = %[[VAL_96]] to %[[VAL_97]] step %[[VAL_95]] unordered iter_args(%[[VAL_100:.*]] = %[[VAL_94]]) -> (!fir.array<?x!fir.char<1,?>>) {
! CHECK:             %[[VAL_101:.*]] = fir.array_access %[[VAL_12]], %[[VAL_99]] typeparams %[[VAL_3]]#1 : (!fir.array<20x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_102:.*]] = fir.array_access %[[VAL_100]], %[[VAL_99]] typeparams %[[VAL_9]] : (!fir.array<?x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_103:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:             %[[VAL_104:.*]] = arith.cmpi slt, %[[VAL_103]], %[[VAL_3]]#1 : index
! CHECK:             %[[VAL_105:.*]] = arith.select %[[VAL_104]], %[[VAL_103]], %[[VAL_3]]#1 : index
! CHECK:             %[[VAL_106:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_107:.*]] = fir.convert %[[VAL_105]] : (index) -> i64
! CHECK:             %[[VAL_108:.*]] = arith.muli %[[VAL_106]], %[[VAL_107]] : i64
! CHECK:             %[[VAL_109:.*]] = arith.constant false
! CHECK:             %[[VAL_110:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_111:.*]] = fir.convert %[[VAL_101]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:             fir.call @llvm.memmove.p0.p0.i64(%[[VAL_110]], %[[VAL_111]], %[[VAL_108]], %[[VAL_109]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:             %[[VAL_112:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_113:.*]] = arith.subi %[[VAL_9]], %[[VAL_112]] : i32
! CHECK:             %[[VAL_114:.*]] = arith.constant 32 : i8
! CHECK:             %[[VAL_115:.*]] = fir.undefined !fir.char<1>
! CHECK:             %[[VAL_116:.*]] = fir.insert_value %[[VAL_115]], %[[VAL_114]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:             %[[VAL_117:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_118:.*]] = fir.convert %[[VAL_113]] : (i32) -> index
! CHECK:             fir.do_loop %[[VAL_119:.*]] = %[[VAL_105]] to %[[VAL_118]] step %[[VAL_117]] {
! CHECK:               %[[VAL_120:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_121:.*]] = fir.coordinate_of %[[VAL_120]], %[[VAL_119]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_116]] to %[[VAL_121]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:             %[[VAL_122:.*]] = fir.array_amend %[[VAL_100]], %[[VAL_102]] : (!fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<?x!fir.char<1,?>>
! CHECK:             fir.result %[[VAL_122]] : !fir.array<?x!fir.char<1,?>>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_94]], %[[VAL_123:.*]] to %[[VAL_92]] typeparams %[[VAL_9]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, i32
! CHECK:           fir.result %[[VAL_90]], %[[VAL_92]] : i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_124:.*]]#0 {
! CHECK:           %[[VAL_125:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           fir.if %[[VAL_17]] {
! CHECK:             fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:           }
! CHECK:           %[[VAL_126:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_127:.*]] = fir.embox %[[VAL_124]]#1(%[[VAL_126]]) typeparams %[[VAL_125]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
! CHECK:           fir.store %[[VAL_127]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
  x = c
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_derived_with_init
subroutine test_derived_with_init(x, y)
  type t 
    integer, allocatable :: a(:)
  end type                                                                                     
  type(t), allocatable :: x                                                                    
  type(t) :: y                                                                                 
  ! The allocatable component of `x` need to be initialized
  ! during the automatic allocation (setting its rank and allocation
  ! status) before it is assigned with the component of `y` 
  x = y
! CHECK:  fir.if %{{.*}} {
! CHECK:    %[[VAL_11:.*]] = fir.allocmem !fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {uniq_name = ".auto.alloc"}
! CHECK:    %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
! CHECK:    %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.box<none>
! CHECK:    fir.call @_FortranAInitialize(%[[VAL_15]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:    fir.result %[[VAL_11]] : !fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! CHECK:  } else {
! CHECK:    fir.result %{{.*}} : !fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_vector_subscript(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"},
! CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"},
! CHECK-SAME: %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "v"}) {
subroutine test_vector_subscript(x, y, v)
  ! Test that the new shape is computed correctly in presence of
  ! vector subscripts on the RHS and that it is used to allocate
  ! the new storage and to drive the implicit loop.
  integer, allocatable :: x(:)
  integer :: y(:), v(:)
  x = y(v)
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_6]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]]#1, %[[VAL_5]]#1 : index
! CHECK:         %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_5]]#1, %[[VAL_7]]#1 : index
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:         %[[VAL_17:.*]]:2 = fir.if %[[VAL_16]] -> (i1, !fir.heap<!fir.array<?xi32>>) {
! CHECK:           %[[VAL_18:.*]] = arith.constant false
! CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_20]]#1, %[[VAL_10]] : index
! CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_21]], %[[VAL_18]] : i1
! CHECK:           %[[VAL_23:.*]] = fir.if %[[VAL_22]] -> (!fir.heap<!fir.array<?xi32>>) {
! CHECK:             %[[VAL_24:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_25:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_26:.*]] = fir.array_load %[[VAL_24]](%[[VAL_25]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
! CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_29:.*]] = arith.subi %[[VAL_10]], %[[VAL_27]] : index
! CHECK:             %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_26]]) -> (!fir.array<?xi32>) {
! CHECK:               %[[VAL_33:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_31]] : (!fir.array<?xi32>, index) -> i32
! CHECK:               %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
! CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_3]] : index
! CHECK:               %[[VAL_36:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_35]] : (!fir.array<?xi32>, index) -> i32
! CHECK:               %[[VAL_37:.*]] = fir.array_update %[[VAL_32]], %[[VAL_36]], %[[VAL_31]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:               fir.result %[[VAL_37]] : !fir.array<?xi32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_26]], %[[VAL_38:.*]] to %[[VAL_24]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.result %[[VAL_24]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           } else {
! CHECK:             %[[VAL_39:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_40:.*]] = fir.array_load %[[VAL_13]](%[[VAL_39]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
! CHECK:             %[[VAL_41:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_42:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_43:.*]] = arith.subi %[[VAL_10]], %[[VAL_41]] : index
! CHECK:             %[[VAL_44:.*]] = fir.do_loop %[[VAL_45:.*]] = %[[VAL_42]] to %[[VAL_43]] step %[[VAL_41]] unordered iter_args(%[[VAL_46:.*]] = %[[VAL_40]]) -> (!fir.array<?xi32>) {
! CHECK:               %[[VAL_47:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_45]] : (!fir.array<?xi32>, index) -> i32
! CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> index
! CHECK:               %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_3]] : index
! CHECK:               %[[VAL_50:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_49]] : (!fir.array<?xi32>, index) -> i32
! CHECK:               %[[VAL_51:.*]] = fir.array_update %[[VAL_46]], %[[VAL_50]], %[[VAL_45]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:               fir.result %[[VAL_51]] : !fir.array<?xi32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_40]], %[[VAL_52:.*]] to %[[VAL_13]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.result %[[VAL_13]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_22]], %[[VAL_53:.*]] : i1, !fir.heap<!fir.array<?xi32>>
! CHECK:         } else {
! CHECK:           %[[VAL_54:.*]] = arith.constant true
! CHECK:           %[[VAL_55:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_56:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_57:.*]] = fir.array_load %[[VAL_55]](%[[VAL_56]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
! CHECK:           %[[VAL_58:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_59:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_60:.*]] = arith.subi %[[VAL_10]], %[[VAL_58]] : index
! CHECK:           %[[VAL_61:.*]] = fir.do_loop %[[VAL_62:.*]] = %[[VAL_59]] to %[[VAL_60]] step %[[VAL_58]] unordered iter_args(%[[VAL_63:.*]] = %[[VAL_57]]) -> (!fir.array<?xi32>) {
! CHECK:             %[[VAL_64:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_62]] : (!fir.array<?xi32>, index) -> i32
! CHECK:             %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i32) -> index
! CHECK:             %[[VAL_66:.*]] = arith.subi %[[VAL_65]], %[[VAL_3]] : index
! CHECK:             %[[VAL_67:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_66]] : (!fir.array<?xi32>, index) -> i32
! CHECK:             %[[VAL_68:.*]] = fir.array_update %[[VAL_63]], %[[VAL_67]], %[[VAL_62]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:             fir.result %[[VAL_68]] : !fir.array<?xi32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_57]], %[[VAL_69:.*]] to %[[VAL_55]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.result %[[VAL_54]], %[[VAL_55]] : i1, !fir.heap<!fir.array<?xi32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_70:.*]]#0 {
! CHECK:           fir.if %[[VAL_16]] {
! CHECK:             fir.freemem %[[VAL_13]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           %[[VAL_71:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_72:.*]] = fir.embox %[[VAL_70]]#1(%[[VAL_71]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_72]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_both_sides_with_elemental_call(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
subroutine test_both_sides_with_elemental_call(x)
  interface
     elemental real function elt(x)
       real, intent(in) :: x
     end function elt
  end interface
  real, allocatable  :: x(:)
  x = elt(x)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_5:.*]] = fir.shape_shift %[[VAL_3]]#0, %[[VAL_3]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?xf32>>) -> i64
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:         %[[VAL_11:.*]]:2 = fir.if %[[VAL_10]] -> (i1, !fir.heap<!fir.array<?xf32>>) {
! CHECK:           %[[VAL_12:.*]] = arith.constant false
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_13]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_14]]#1, %[[VAL_3]]#1 : index
! CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_15]], %[[VAL_12]] : i1
! CHECK:           %[[VAL_17:.*]] = fir.if %[[VAL_16]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_18:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_3]]#1 {uniq_name = ".auto.alloc"}
! CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_20:.*]] = fir.array_load %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_22:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_21]] : index
! CHECK:             %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_20]]) -> (!fir.array<?xf32>) {
! CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_25]], %[[VAL_3]]#0 : index
! CHECK:               %[[VAL_28:.*]] = fir.array_coor %[[VAL_4]](%[[VAL_5]]) %[[VAL_27]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
! CHECK:               %[[VAL_29:.*]] = fir.call @_QPelt(%[[VAL_28]]) {{.*}}: (!fir.ref<f32>) -> f32
! CHECK:               %[[VAL_30:.*]] = fir.array_update %[[VAL_26]], %[[VAL_29]], %[[VAL_25]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:               fir.result %[[VAL_30]] : !fir.array<?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_20]], %[[VAL_31:.*]] to %[[VAL_18]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:             fir.result %[[VAL_18]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_32:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_33:.*]] = fir.array_load %[[VAL_7]](%[[VAL_32]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:             %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_35:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_34]] : index
! CHECK:             %[[VAL_37:.*]] = fir.do_loop %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_36]] step %[[VAL_34]] unordered iter_args(%[[VAL_39:.*]] = %[[VAL_33]]) -> (!fir.array<?xf32>) {
! CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_38]], %[[VAL_3]]#0 : index
! CHECK:               %[[VAL_41:.*]] = fir.array_coor %[[VAL_4]](%[[VAL_5]]) %[[VAL_40]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
! CHECK:               %[[VAL_42:.*]] = fir.call @_QPelt(%[[VAL_41]]) {{.*}}: (!fir.ref<f32>) -> f32
! CHECK:               %[[VAL_43:.*]] = fir.array_update %[[VAL_39]], %[[VAL_42]], %[[VAL_38]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:               fir.result %[[VAL_43]] : !fir.array<?xf32>
! CHECK:             }
! CHECK:             fir.array_merge_store %[[VAL_33]], %[[VAL_44:.*]] to %[[VAL_7]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:             fir.result %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           }
! CHECK:           fir.result %[[VAL_16]], %[[VAL_45:.*]] : i1, !fir.heap<!fir.array<?xf32>>
! CHECK:         } else {
! CHECK:           %[[VAL_46:.*]] = arith.constant true
! CHECK:           %[[VAL_47:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_3]]#1 {uniq_name = ".auto.alloc"}
! CHECK:           %[[VAL_48:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_49:.*]] = fir.array_load %[[VAL_47]](%[[VAL_48]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:           %[[VAL_50:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_51:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_52:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_50]] : index
! CHECK:           %[[VAL_53:.*]] = fir.do_loop %[[VAL_54:.*]] = %[[VAL_51]] to %[[VAL_52]] step %[[VAL_50]] unordered iter_args(%[[VAL_55:.*]] = %[[VAL_49]]) -> (!fir.array<?xf32>) {
! CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_54]], %[[VAL_3]]#0 : index
! CHECK:             %[[VAL_57:.*]] = fir.array_coor %[[VAL_4]](%[[VAL_5]]) %[[VAL_56]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
! CHECK:             %[[VAL_58:.*]] = fir.call @_QPelt(%[[VAL_57]]) {{.*}}: (!fir.ref<f32>) -> f32
! CHECK:             %[[VAL_59:.*]] = fir.array_update %[[VAL_55]], %[[VAL_58]], %[[VAL_54]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:             fir.result %[[VAL_59]] : !fir.array<?xf32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_49]], %[[VAL_60:.*]] to %[[VAL_47]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.result %[[VAL_46]], %[[VAL_47]] : i1, !fir.heap<!fir.array<?xf32>>
! CHECK:         }
! CHECK:         fir.if %[[VAL_61:.*]]#0 {
! CHECK:           fir.if %[[VAL_10]] {
! CHECK:             fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_62:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_63:.*]] = fir.embox %[[VAL_61]]#1(%[[VAL_62]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_63]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         }
! CHECK:         return
! CHECK:       }
end subroutine

! CHECK: fir.global linkonce @[[error_message]] constant : !fir.char<1,76> {
! CHECK:   %[[msg:.*]] = fir.string_lit "array left hand side must be allocated when the right hand side is a scalar\00"(76) : !fir.char<1,76>
! CHECK:   fir.has_value %[[msg:.*]] : !fir.char<1,76>
! CHECK: }

end module

!  use alloc_assign
!  real :: y(2, 3) = reshape([1,2,3,4,5,6], [2,3])
!  real, allocatable :: x (:, :)
!  allocate(x(2,2))
!  call test_with_lbounds(x, y) 
!  print *, x(10, 20)
!  print *, x
!end
