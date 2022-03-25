! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmvbits_test(
function mvbits_test(from, frompos, len, to, topos)
  ! CHECK: %[[result:.*]] = fir.alloca i32 {bindc_name = "mvbits_test"
  ! CHECK-DAG: %[[from:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[frompos:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[len:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG: %[[to:.*]] = fir.load %arg3 : !fir.ref<i32>
  ! CHECK-DAG: %[[topos:.*]] = fir.load %arg4 : !fir.ref<i32>
  integer :: from, frompos, len, to, topos
  integer :: mvbits_test
  ! CHECK: %[[VAL_11:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_12:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_13:.*]] = arith.constant 32 : i32
  ! CHECK: %[[VAL_14:.*]] = arith.subi %[[VAL_13]], %[[len]] : i32
  ! CHECK: %[[VAL_15:.*]] = arith.shrui %[[VAL_12]], %[[VAL_14]] : i32
  ! CHECK: %[[VAL_16:.*]] = arith.shli %[[VAL_15]], %[[topos]] : i32
  ! CHECK: %[[VAL_17:.*]] = arith.xori %[[VAL_16]], %[[VAL_12]] : i32
  ! CHECK: %[[VAL_18:.*]] = arith.andi %[[VAL_17]], %[[to]] : i32
  ! CHECK: %[[VAL_19:.*]] = arith.shrui %[[from]], %[[frompos]] : i32
  ! CHECK: %[[VAL_20:.*]] = arith.andi %[[VAL_19]], %[[VAL_15]] : i32
  ! CHECK: %[[VAL_21:.*]] = arith.shli %[[VAL_20]], %[[topos]] : i32
  ! CHECK: %[[VAL_22:.*]] = arith.ori %[[VAL_18]], %[[VAL_21]] : i32
  ! CHECK: %[[VAL_23:.*]] = arith.cmpi eq, %[[len]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[to]], %[[VAL_22]] : i32
  ! CHECK: fir.store %[[VAL_24]] to %arg3 : !fir.ref<i32>
  ! CHECK: %[[VAL_25:.*]] = fir.load %arg3 : !fir.ref<i32>
  ! CHECK: fir.store %[[VAL_25]] to %[[result]] : !fir.ref<i32>
  call mvbits(from, frompos, len, to, topos)
  ! CHECK: %[[VAL_26:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_26]] : i32
  mvbits_test = to
end

! CHECK-LABEL: func @_QPmvbits_array_test(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_4:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_6]]#1, %[[VAL_11]] : index
! CHECK:         fir.do_loop %[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_11]] {
! CHECK:           %[[VAL_15:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_14]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_16]] : index
! CHECK:           %[[VAL_18:.*]] = fir.array_coor %[[VAL_3]] %[[VAL_17]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_21:.*]] = arith.constant -1 : i32
! CHECK:           %[[VAL_22:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_24:.*]] = arith.shrui %[[VAL_21]], %[[VAL_23]] : i32
! CHECK:           %[[VAL_25:.*]] = arith.shli %[[VAL_24]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_26:.*]] = arith.xori %[[VAL_25]], %[[VAL_21]] : i32
! CHECK:           %[[VAL_27:.*]] = arith.andi %[[VAL_26]], %[[VAL_19]] : i32
! CHECK:           %[[VAL_28:.*]] = arith.shrui %[[VAL_15]], %[[VAL_8]] : i32
! CHECK:           %[[VAL_29:.*]] = arith.andi %[[VAL_28]], %[[VAL_24]] : i32
! CHECK:           %[[VAL_30:.*]] = arith.shli %[[VAL_29]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_31:.*]] = arith.ori %[[VAL_27]], %[[VAL_30]] : i32
! CHECK:           %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_20]] : i32
! CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_19]], %[[VAL_31]] : i32
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_18]] : !fir.ref<i32>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine mvbits_array_test(from, frompos, len, to, topos)
  integer :: from(:), frompos, len, to(:), topos

  call mvbits(from, frompos, len, to, topos)
end subroutine
