! Check that the implied-do index value is converted to proper type.
! RUN: bbc -emit-fir -o - %s | FileCheck %s

function test1(k)
  integer*1 :: k
  integer*1 :: test1(4)
  test1 = ([(i*k, integer(8)::i=1,4)])
end function test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i8> {fir.bindc_name = "k"}) -> !fir.array<4xi8> {
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[ARG0_D:.*]] = fir.declare %[[ARG0]]
! CHECK:           %[[RES:.*]] = fir.alloca !fir.array<4xi8>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C4]]
! CHECK:           %[[RES_D:.*]] = fir.declare %[[RES]](%[[SHAPE]])
! CHECK:           fir.do_loop %[[IDX:.*]] = %[[C1]] to %[[C4]] step %[[C1]] unordered {
! CHECK:             %[[IDX_I64:.*]] = fir.convert %[[IDX]] : (index) -> i64
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]] : !fir.ref<i8>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i8) -> i64
! CHECK:             %[[PROD:.*]] = arith.muli %[[IDX_I64]], %[[VAL_K_I64]] : i64
! CHECK:             %[[PROD_NO:.*]] = fir.no_reassoc %[[PROD]] : i64
! CHECK:             %[[PROD_I8:.*]] = fir.convert %[[PROD_NO]] : (i64) -> i8
! CHECK:             %[[ADDR:.*]] = fir.array_coor %[[RES_D]](%[[SHAPE]]) %[[IDX]]
! CHECK:             fir.store %[[PROD_I8]] to %[[ADDR]] : !fir.ref<i8>
! CHECK:           }
! CHECK:         }

function test2(k)
  integer*2 :: k
  integer*2 :: test2(4)
  test2 = ([(i*k, integer(8)::i=1,4)])
end function test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i16> {fir.bindc_name = "k"}) -> !fir.array<4xi16> {
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[ARG0_D:.*]] = fir.declare %[[ARG0]]
! CHECK:           %[[RES:.*]] = fir.alloca !fir.array<4xi16>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C4]]
! CHECK:           %[[RES_D:.*]] = fir.declare %[[RES]](%[[SHAPE]])
! CHECK:           fir.do_loop %[[IDX:.*]] = %[[C1]] to %[[C4]] step %[[C1]] unordered {
! CHECK:             %[[IDX_I64:.*]] = fir.convert %[[IDX]] : (index) -> i64
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]] : !fir.ref<i16>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i16) -> i64
! CHECK:             %[[PROD:.*]] = arith.muli %[[IDX_I64]], %[[VAL_K_I64]] : i64
! CHECK:             %[[PROD_NO:.*]] = fir.no_reassoc %[[PROD]] : i64
! CHECK:             %[[PROD_I16:.*]] = fir.convert %[[PROD_NO]] : (i64) -> i16
! CHECK:             %[[ADDR:.*]] = fir.array_coor %[[RES_D]](%[[SHAPE]]) %[[IDX]]
! CHECK:             fir.store %[[PROD_I16]] to %[[ADDR]] : !fir.ref<i16>
! CHECK:           }
! CHECK:         }

function test3(k)
  integer*4 :: k
  integer*4 :: test3(4)
  test3 = ([(i*k, integer(8)::i=1,4)])
end function test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "k"}) -> !fir.array<4xi32> {
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[ARG0_D:.*]] = fir.declare %[[ARG0]]
! CHECK:           %[[RES:.*]] = fir.alloca !fir.array<4xi32>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C4]]
! CHECK:           %[[RES_D:.*]] = fir.declare %[[RES]](%[[SHAPE]])
! CHECK:           fir.do_loop %[[IDX:.*]] = %[[C1]] to %[[C4]] step %[[C1]] unordered {
! CHECK:             %[[IDX_I64:.*]] = fir.convert %[[IDX]] : (index) -> i64
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]] : !fir.ref<i32>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i32) -> i64
! CHECK:             %[[PROD:.*]] = arith.muli %[[IDX_I64]], %[[VAL_K_I64]] : i64
! CHECK:             %[[PROD_NO:.*]] = fir.no_reassoc %[[PROD]] : i64
! CHECK:             %[[PROD_I32:.*]] = fir.convert %[[PROD_NO]] : (i64) -> i32
! CHECK:             %[[ADDR:.*]] = fir.array_coor %[[RES_D]](%[[SHAPE]]) %[[IDX]]
! CHECK:             fir.store %[[PROD_I32]] to %[[ADDR]] : !fir.ref<i32>
! CHECK:           }
! CHECK:         }

function test4(k)
  integer*8 :: k
  integer*8 :: test4(4)
  test4 = ([(i*k, integer(8)::i=1,4)])
end function test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i64> {fir.bindc_name = "k"}) -> !fir.array<4xi64> {
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[ARG0_D:.*]] = fir.declare %[[ARG0]]
! CHECK:           %[[RES:.*]] = fir.alloca !fir.array<4xi64>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C4]]
! CHECK:           %[[RES_D:.*]] = fir.declare %[[RES]](%[[SHAPE]])
! CHECK:           fir.do_loop %[[IDX:.*]] = %[[C1]] to %[[C4]] step %[[C1]] unordered {
! CHECK:             %[[IDX_I64:.*]] = fir.convert %[[IDX]] : (index) -> i64
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]] : !fir.ref<i64>
! CHECK:             %[[PROD:.*]] = arith.muli %[[IDX_I64]], %[[VAL_K]] : i64
! CHECK:             %[[PROD_NO:.*]] = fir.no_reassoc %[[PROD]] : i64
! CHECK:             %[[ADDR:.*]] = fir.array_coor %[[RES_D]](%[[SHAPE]]) %[[IDX]]
! CHECK:             fir.store %[[PROD_NO]] to %[[ADDR]] : !fir.ref<i64>
! CHECK:           }
! CHECK:         }
