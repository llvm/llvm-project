! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) -> !fir.array<1x!fir.logical<4>> {
! CHECK-DAG:         %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK-DAG:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK-DAG:         %[[VAL_3:.*]] = arith.constant 0 : i32
! CHECK-DAG:         %[[VAL_4:.*]] = arith.constant 8 : index
! CHECK-DAG:         %[[VAL_5:.*]] = arith.constant 8 : i64
! CHECK:         %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<1x!fir.char<1,12>>>
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.array<1x!fir.logical<4>> {bindc_name = "test", uniq_name = "_QFtestEtest"}
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.slice %[[VAL_1]], %[[VAL_1]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_11:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.array<1x!fir.char<1,8>>>
! CHECK:         br ^bb1(%[[VAL_2]], %[[VAL_1]] : index, index)
! CHECK:       ^bb1(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
! CHECK:         %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_2]] : index
! CHECK:         cond_br %[[VAL_14]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
! CHECK:         %[[VAL_16:.*]] = fir.array_coor %[[VAL_7]](%[[VAL_9]]) {{\[}}%[[VAL_10]]] %[[VAL_15]] : (!fir.ref<!fir.array<1x!fir.char<1,12>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,12>>
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<!fir.array<12x!fir.char<1>>>
! CHECK:         %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_2]] : (!fir.ref<!fir.array<12x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_20:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_9]]) %[[VAL_15]] : (!fir.ref<!fir.array<1x!fir.char<1,8>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,8>>
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_24:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_5]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
! CHECK:         %[[VAL_25:.*]] = arith.cmpi eq, %[[VAL_24]], %[[VAL_3]] : i32
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i1) -> !fir.logical<4>
! CHECK:         %[[VAL_27:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_9]]) %[[VAL_15]] : (!fir.ref<!fir.array<1x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:         fir.store %[[VAL_26]] to %[[VAL_27]] : !fir.ref<!fir.logical<4>>
! CHECK:         %[[VAL_28:.*]] = arith.subi %[[VAL_13]], %[[VAL_1]] : index
! CHECK:         br ^bb1(%[[VAL_15]], %[[VAL_28]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_29:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.array<1x!fir.logical<4>>>
! CHECK:         return %[[VAL_29]] : !fir.array<1x!fir.logical<4>>
! CHECK:       }


function test(C)
  logical :: test(1)
  character*12  C(1)

  test = C(1:1)(1:8) == (/'ABCDabcd'/) 
end function test
