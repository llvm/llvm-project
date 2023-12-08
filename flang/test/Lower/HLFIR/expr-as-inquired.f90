! Test lowering to HLFIR of the intrinsic lowering framework
! "asInquired" option.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_isAllocated(x, l)
  logical :: l
  real, allocatable :: x(:)
  l = allocated(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_isallocated(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}}  {{.*}}El
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, {{.*}}Ex"
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.heap<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:  hlfir.assign %[[VAL_9]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:  return
! CHECK:  }

subroutine test_lbound(x, n)
  integer :: n
  real :: x(2:, 3:)
  n = lbound(x, dim=n)
end subroutine
! CHECK-LABEL: func.func @_QPtest_lbound(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}  {{.*}}En
! CHECK:  %[[VAL_5:.*]] = arith.constant 2 : i64
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:  %[[VAL_7:.*]] = arith.constant 3 : i64
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}})  {{.*}}Ex
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_12:.*]] = fir.shift %[[VAL_6]], %[[VAL_8]] : (index, index) -> !fir.shift<2>
! CHECK:  %[[VAL_13:.*]] = fir.rebox %[[VAL_10]]#1(%[[VAL_12]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:  %[[VAL_18:.*]] = fir.call @_FortranALboundDim(%[[VAL_16]],
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> i32
! CHECK:  hlfir.assign %[[VAL_19]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
