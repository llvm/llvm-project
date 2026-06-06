! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmvbits_test(
function mvbits_test(from, frompos, len, to, topos)
  integer :: from, frompos, len, to, topos
  integer :: mvbits_test
  ! CHECK: %[[from_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 {uniq_name = "_QFmvbits_testEfrom"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[frompos_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFmvbits_testEfrompos"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[len_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 3 {uniq_name = "_QFmvbits_testElen"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[to_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 4 {uniq_name = "_QFmvbits_testEto"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[topos_decl:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 5 {uniq_name = "_QFmvbits_testEtopos"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-DAG: %[[from:.*]] = fir.load %[[from_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[frompos:.*]] = fir.load %[[frompos_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[len:.*]] = fir.load %[[len_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[topos:.*]] = fir.load %[[topos_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[to:.*]] = fir.load %[[to_decl]]#0 : !fir.ref<i32>
  ! CHECK: %[[VAL_0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_1:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_2:.*]] = arith.constant 32 : i32
  ! CHECK: %[[VAL_3:.*]] = arith.subi %[[VAL_2]], %[[len]] : i32
  ! CHECK: %[[VAL_4:.*]] = arith.shrui %[[VAL_1]], %[[VAL_3]] : i32
  ! CHECK: %[[VAL_5:.*]] = arith.shli %[[VAL_4]], %[[topos]] : i32
  ! CHECK: %[[VAL_6:.*]] = arith.xori %[[VAL_5]], %[[VAL_1]] : i32
  ! CHECK: %[[VAL_7:.*]] = arith.andi %[[VAL_6]], %[[to]] : i32
  ! CHECK: %[[VAL_8:.*]] = arith.shrui %[[from]], %[[frompos]] : i32
  ! CHECK: %[[VAL_9:.*]] = arith.andi %[[VAL_8]], %[[VAL_4]] : i32
  ! CHECK: %[[VAL_10:.*]] = arith.shli %[[VAL_9]], %[[topos]] : i32
  ! CHECK: %[[VAL_11:.*]] = arith.ori %[[VAL_7]], %[[VAL_10]] : i32
  ! CHECK: %[[VAL_12:.*]] = arith.cmpi eq, %[[len]], %[[VAL_0]] : i32
  ! CHECK: %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[to]], %[[VAL_11]] : i32
  ! CHECK: fir.store %[[VAL_13]] to %[[to_decl]]#0 : !fir.ref<i32>
  call mvbits(from, frompos, len, to, topos)
  mvbits_test = to
end

! CHECK-LABEL: func @_QPmvbits_array_test(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_4:.*]]: !fir.ref<i32>{{.*}}) {
subroutine mvbits_array_test(from, frompos, len, to, topos)
  integer :: from(:), frompos, len, to(:), topos

  ! CHECK: %[[from_decl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFmvbits_array_testEfrom"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  ! CHECK: %[[to_decl:.*]]:2 = hlfir.declare %[[VAL_3]] {{.*}} {uniq_name = "_QFmvbits_array_testEto"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[from_decl]]#0, %[[c0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  ! CHECK: %[[c1:.*]] = arith.constant 1 : index
  ! CHECK: fir.do_loop %[[i:.*]] = %[[c1]] to %[[dims]]#1 step %[[c1]] {
  ! CHECK:   %[[from_elem:.*]] = hlfir.designate %[[from_decl]]#0 (%[[i]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  ! CHECK:   %[[from_val:.*]] = fir.load %[[from_elem]] : !fir.ref<i32>
  ! CHECK:   %[[to_elem:.*]] = hlfir.designate %[[to_decl]]#0 (%[[i]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  ! CHECK:   %[[to_val:.*]] = fir.load %[[to_elem]] : !fir.ref<i32>
  ! CHECK:   %[[VAL_0:.*]] = arith.constant 0 : i32
  ! CHECK:   %[[VAL_1:.*]] = arith.constant -1 : i32
  ! CHECK:   %[[VAL_2:.*]] = arith.constant 32 : i32
  ! CHECK:   arith.subi
  ! CHECK:   arith.shrui
  ! CHECK:   arith.shli
  ! CHECK:   arith.xori
  ! CHECK:   arith.andi
  ! CHECK:   arith.shrui
  ! CHECK:   arith.andi
  ! CHECK:   arith.shli
  ! CHECK:   arith.ori
  ! CHECK:   %[[cmp:.*]] = arith.cmpi eq
  ! CHECK:   %[[sel:.*]] = arith.select %[[cmp]]
  ! CHECK:   fir.store %[[sel]] to %[[to_elem]] : !fir.ref<i32>
  ! CHECK: }
  ! CHECK: return

  call mvbits(from, frompos, len, to, topos)
end subroutine
