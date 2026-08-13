! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

subroutine test_atomic
  use iso_fortran_env, only: atomic_int_kind
  implicit none

  integer(atomic_int_kind), save :: atom[*]

  atom = 10

  call atomic_add(atom, 5)

  call atomic_and(atom, 12)

  call atomic_or(atom, 3)
  
  call atomic_xor(atom, 5)

end subroutine test_atomic

! COARRAY-LABEL: func.func @_QPtest_atomic() {
! COARRAY:  %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY:  %[[VAL_1:.*]] = fir.address_of(@_QFtest_atomicEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_atomicEatom")
! COARRAY:  %[[VAL_3:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
! COARRAY:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter>
! COARRAY:  %[[VAL_5:.*]] = arith.constant 10 : i64
! COARRAY:  %[[VAL_6:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  hlfir.assign %[[VAL_5]] to %[[VAL_7]] : i64, !fir.heap<i64>
! COARRAY:  %[[VAL_8:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_10:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_add atom %[[VAL_9]] value %[[VAL_8]] stat %[[VAL_10]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_11:.*]] = arith.constant 12 : i32
! COARRAY:  %[[VAL_12:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_13:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_and atom %[[VAL_12]] value %[[VAL_11]] stat %[[VAL_13]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_14:.*]] = arith.constant 3 : i32
! COARRAY:  %[[VAL_15:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_16:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_or atom %[[VAL_15]] value %[[VAL_14]] stat %[[VAL_16]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_17:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_18:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_19:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_xor atom %[[VAL_18]] value %[[VAL_17]] stat %[[VAL_19]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
! COARRAY:  return

subroutine test_atomic_with_coindex
  use iso_fortran_env, only: atomic_int_kind
  implicit none

  integer(atomic_int_kind), save :: atom[*]
  integer :: i = 1

  atom = 10

  call atomic_add(atom[1], 5)

  call atomic_and(atom[i], 12)

  call atomic_or(atom[1], 3)
  
  call atomic_xor(atom[i], 5)

end subroutine test_atomic_with_coindex

! COARRAY-LABEL: func.func @_QPtest_atomic_with_coindex() {
! COARRAY:  %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY:  %[[VAL_1:.*]] = fir.address_of(@_QFtest_atomic_with_coindexEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_atomic_with_coindexEatom")
! COARRAY:  %[[VAL_3:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
! COARRAY:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter>
! COARRAY:  %[[VAL_5:.*]] = fir.address_of(@_QFtest_atomic_with_coindexEi) : !fir.ref<i32>
! COARRAY:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFtest_atomic_with_coindexEi")
! COARRAY:  %[[VAL_7:.*]] = arith.constant 10 : i64
! COARRAY:  %[[VAL_8:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  hlfir.assign %[[VAL_7]] to %[[VAL_9]] : i64, !fir.heap<i64>
! COARRAY:  %[[VAL_10:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_12:.*]] = arith.constant 1 : i64
! COARRAY:  %[[VAL_13:.*]] = hlfir.designate %[[VAL_11]] [%[[VAL_12]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_14:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_15:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_16:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_add atom %[[VAL_15]][%[[VAL_12]]] value %[[VAL_14]] stat %[[VAL_16]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_17:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_19:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! COARRAY:  %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! COARRAY:  %[[VAL_21:.*]] = hlfir.designate %[[VAL_18]] [%[[VAL_20]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_22:.*]] = arith.constant 12 : i32
! COARRAY:  %[[VAL_23:.*]] = fir.embox %[[VAL_21]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_24:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_and atom %[[VAL_23]][%[[VAL_20]]] value %[[VAL_22]] stat %[[VAL_24]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_25:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_26:.*]] = fir.box_addr %[[VAL_25]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_27:.*]] = arith.constant 1 : i64
! COARRAY:  %[[VAL_28:.*]] = hlfir.designate %[[VAL_26]] [%[[VAL_27]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_29:.*]] = arith.constant 3 : i32
! COARRAY:  %[[VAL_30:.*]] = fir.embox %[[VAL_28]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_31:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_or atom %[[VAL_30]][%[[VAL_27]]] value %[[VAL_29]] stat %[[VAL_31]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_32:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_33:.*]] = fir.box_addr %[[VAL_32]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_34:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! COARRAY:  %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! COARRAY:  %[[VAL_36:.*]] = hlfir.designate %[[VAL_33]] [%[[VAL_35]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_37:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_38:.*]] = fir.embox %[[VAL_36]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_39:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_xor atom %[[VAL_38]][%[[VAL_35]]] value %[[VAL_37]] stat %[[VAL_39]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
! COARRAY:  return

