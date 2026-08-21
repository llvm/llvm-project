! Check that the implied-do index value is converted to proper type.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

function test1(k)
  integer*1 :: k
  integer*1 :: test1(4)
  test1 = ([(i*k, integer(8)::i=1,4)])
end function test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i8> {fir.bindc_name = "k"}) -> !fir.array<4xi8> {
! CHECK:           %[[ARG0_D:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]]#0 : !fir.ref<i8>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i8) -> i64
! CHECK:             arith.muli {{.*}}, %[[VAL_K_I64]] : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             hlfir.no_reassoc {{.*}} : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi8> {
! CHECK:             fir.convert {{.*}} : (i64) -> i8
! CHECK:           }
! CHECK:           hlfir.assign {{.*}} to {{.*}} : !hlfir.expr<4xi8>, !fir.ref<!fir.array<4xi8>>
! CHECK:         }

function test2(k)
  integer*2 :: k
  integer*2 :: test2(4)
  test2 = ([(i*k, integer(8)::i=1,4)])
end function test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i16> {fir.bindc_name = "k"}) -> !fir.array<4xi16> {
! CHECK:           %[[ARG0_D:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]]#0 : !fir.ref<i16>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i16) -> i64
! CHECK:             arith.muli {{.*}}, %[[VAL_K_I64]] : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             hlfir.no_reassoc {{.*}} : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi16> {
! CHECK:             fir.convert {{.*}} : (i64) -> i16
! CHECK:           }
! CHECK:           hlfir.assign {{.*}} to {{.*}} : !hlfir.expr<4xi16>, !fir.ref<!fir.array<4xi16>>
! CHECK:         }

function test3(k)
  integer*4 :: k
  integer*4 :: test3(4)
  test3 = ([(i*k, integer(8)::i=1,4)])
end function test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "k"}) -> !fir.array<4xi32> {
! CHECK:           %[[ARG0_D:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_K_I64:.*]] = fir.convert %[[VAL_K]] : (i32) -> i64
! CHECK:             arith.muli {{.*}}, %[[VAL_K_I64]] : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             hlfir.no_reassoc {{.*}} : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi32> {
! CHECK:             fir.convert {{.*}} : (i64) -> i32
! CHECK:           }
! CHECK:           hlfir.assign {{.*}} to {{.*}} : !hlfir.expr<4xi32>, !fir.ref<!fir.array<4xi32>>
! CHECK:         }

function test4(k)
  integer*8 :: k
  integer*8 :: test4(4)
  test4 = ([(i*k, integer(8)::i=1,4)])
end function test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<i64> {fir.bindc_name = "k"}) -> !fir.array<4xi64> {
! CHECK:           %[[ARG0_D:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             %[[VAL_K:.*]] = fir.load %[[ARG0_D]]#0 : !fir.ref<i64>
! CHECK:             arith.muli {{.*}}, %[[VAL_K]] : i64
! CHECK:           }
! CHECK:           hlfir.elemental {{.*}} -> !hlfir.expr<4xi64> {
! CHECK:             hlfir.no_reassoc {{.*}} : i64
! CHECK:           }
! CHECK:           hlfir.assign {{.*}} to {{.*}} : !hlfir.expr<4xi64>, !fir.ref<!fir.array<4xi64>>
! CHECK:         }
