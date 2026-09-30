! RUN: %flang_fc1 -funsigned -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmvbits_unsigned_test(
subroutine mvbits_unsigned_test(from, frompos, len, to, topos)
  unsigned :: from, to
  integer :: frompos, len, topos
  ! CHECK: %[[from_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmvbits_unsigned_testEfrom"} : (!fir.ref<ui32>, !fir.dscope) -> (!fir.ref<ui32>, !fir.ref<ui32>)
  ! CHECK: %[[frompos_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmvbits_unsigned_testEfrompos"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[len_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmvbits_unsigned_testElen"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[to_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 4 {uniq_name = "_QFmvbits_unsigned_testEto"} : (!fir.ref<ui32>, !fir.dscope) -> (!fir.ref<ui32>, !fir.ref<ui32>)
  ! CHECK: %[[topos_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 5 {uniq_name = "_QFmvbits_unsigned_testEtopos"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-DAG: %[[from:.*]] = fir.load %[[from_decl]]#0 : !fir.ref<ui32>
  ! CHECK-DAG: %[[frompos:.*]] = fir.load %[[frompos_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[len:.*]] = fir.load %[[len_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[topos:.*]] = fir.load %[[topos_decl]]#0 : !fir.ref<i32>
  ! CHECK: %[[to:.*]] = fir.load %[[to_decl]]#0 : !fir.ref<ui32>
  ! CHECK: %[[to_i:.*]] = fir.convert %[[to]] : (ui32) -> i32
  ! CHECK: %[[VAL_0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_2:.*]] = arith.constant 32 : i32
  ! CHECK: %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[len]] : i32
  ! CHECK: %[[VAL_4:.*]] = arith.shrui %[[VAL_1]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_5:.*]] = arith.shli %[[VAL_4]], %[[topos]] : i32
  ! CHECK: %[[VAL_6:.*]] = arith.xori %[[VAL_5]], %[[VAL_1]] : i32
  ! CHECK: %[[VAL_7:.*]] = arith.andi %[[VAL_6]], %[[to_i]] : i32
  ! CHECK: %[[from_i:.*]] = fir.convert %[[from]] : (ui32) -> i32
  ! CHECK: %[[VAL_8:.*]] = arith.shrui %[[from_i]], %[[frompos]] : i32
  ! CHECK: %[[VAL_9:.*]] = arith.andi %[[VAL_8]], %[[VAL_4]] : i32
  ! CHECK: %[[VAL_10:.*]] = arith.shli %[[VAL_9]], %[[topos]] : i32
  ! CHECK: %[[VAL_11:.*]] = arith.ori %[[VAL_7]], %[[VAL_10]] : i32
  ! CHECK: %[[VAL_12:.*]] = arith.cmpi eq, %[[len]], %[[VAL_0]] : i32
  ! CHECK: %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[to_i]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> ui32
  ! CHECK: fir.store %[[VAL_14]] to %[[to_decl]]#0 : !fir.ref<ui32>
  call mvbits(from, frompos, len, to, topos)
end subroutine
