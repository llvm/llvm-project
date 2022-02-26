! Test allocatable assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!            Test simple scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_simple_scalar(
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

! CHECK-LABEL: func @_QPtest_simple_local_scalar() {
subroutine test_simple_local_scalar()
  real, allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.heap<f32> {uniq_name = "_QFtest_simple_local_scalarEx.addr"}
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
