! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

subroutine test_atomic_fetch
  use iso_fortran_env, only: atomic_int_kind
  implicit none

  integer(atomic_int_kind), save :: atom[*]
  integer(kind=atomic_int_kind) :: old_val

  atom = 10

  call atomic_fetch_add(atom, 5, OLD=old_val)

  call atomic_fetch_and(atom, 12, OLD=old_val)

  call atomic_fetch_or(atom, 3, OLD=old_val)
  
  call atomic_fetch_xor(atom, 5, OLD=old_val)

end subroutine test_atomic_fetch

subroutine test_atomic_fetch_with_coindex
  use iso_fortran_env, only: atomic_int_kind
  implicit none

  integer(atomic_int_kind), save :: atom[*]
  integer(kind=atomic_int_kind) :: old_val
  integer :: i = 1

  atom = 10

  call atomic_fetch_add(atom[1], 5, OLD=old_val)

  call atomic_fetch_and(atom[i], 12, OLD=old_val)

  call atomic_fetch_or(atom[1], 3, OLD=old_val)
  
  call atomic_fetch_xor(atom[i], 5, OLD=old_val)

end subroutine test_atomic_fetch_with_coindex

! COARRAY-LABEL: func.func @_QPtest_atomic_fetch() {
! COARRAY:  %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY:  %[[VAL_1:.*]] = fir.address_of(@_QFtest_atomic_fetchEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_atomic_fetchEatom")
! COARRAY:  %[[VAL_3:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
! COARRAY:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter>
! COARRAY:  %[[VAL_5:.*]] = fir.alloca i64 <{bindc_name = "old_val", uniq_name = "_QFtest_atomic_fetchEold_val"}>
! COARRAY:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFtest_atomic_fetchEold_val")
! COARRAY:  %[[VAL_7:.*]] = arith.constant 10 : i64
! COARRAY:  %[[VAL_8:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  hlfir.assign %[[VAL_7]] to %[[VAL_9]] : i64, !fir.heap<i64>
! COARRAY:  %[[VAL_10:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_11:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_12:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_add atom %[[VAL_11]] value %[[VAL_10]] old %[[VAL_6]]#0 stat %[[VAL_12]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_13:.*]] = arith.constant 12 : i32
! COARRAY:  %[[VAL_14:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_15:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_and atom %[[VAL_14]] value %[[VAL_13]] old %[[VAL_6]]#0 stat %[[VAL_15]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_16:.*]] = arith.constant 3 : i32
! COARRAY:  %[[VAL_17:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_18:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_or atom %[[VAL_17]] value %[[VAL_16]] old %[[VAL_6]]#0 stat %[[VAL_18]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_19:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_20:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_21:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_xor atom %[[VAL_20]] value %[[VAL_19]] old %[[VAL_6]]#0 stat %[[VAL_21]] : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  return

! COARRAY-LABEL: func.func @_QPtest_atomic_fetch_with_coindex() {
! COARRAY:  %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY:  %[[VAL_1:.*]] = fir.address_of(@_QFtest_atomic_fetch_with_coindexEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFtest_atomic_fetch_with_coindexEatom")
! COARRAY:  %[[VAL_3:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
! COARRAY:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter>
! COARRAY:  %[[VAL_5:.*]] = fir.address_of(@_QFtest_atomic_fetch_with_coindexEi) : !fir.ref<i32>
! COARRAY:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFtest_atomic_fetch_with_coindexEi")
! COARRAY:  %[[VAL_7:.*]] = fir.alloca i64 <{bindc_name = "old_val", uniq_name = "_QFtest_atomic_fetch_with_coindexEold_val"}>
! COARRAY:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QFtest_atomic_fetch_with_coindexEold_val")
! COARRAY:  %[[VAL_9:.*]] = arith.constant 10 : i64
! COARRAY:  %[[VAL_10:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  hlfir.assign %[[VAL_9]] to %[[VAL_11]] : i64, !fir.heap<i64>
! COARRAY:  %[[VAL_12:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_14:.*]] = arith.constant 1 : i64
! COARRAY:  %[[VAL_15:.*]] = hlfir.designate %[[VAL_13]] [%[[VAL_14]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_16:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_17:.*]] = fir.embox %[[VAL_15]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_18:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_add atom %[[VAL_17]][%[[VAL_14]]] value %[[VAL_16]] old %[[VAL_8]]#0 stat %[[VAL_18]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_19:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_21:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! COARRAY:  %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! COARRAY:  %[[VAL_23:.*]] = hlfir.designate %[[VAL_20]] [%[[VAL_22]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_24:.*]] = arith.constant 12 : i32
! COARRAY:  %[[VAL_25:.*]] = fir.embox %[[VAL_23]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_26:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_and atom %[[VAL_25]][%[[VAL_22]]] value %[[VAL_24]] old %[[VAL_8]]#0 stat %[[VAL_26]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_27:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_28:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_29:.*]] = arith.constant 1 : i64
! COARRAY:  %[[VAL_30:.*]] = hlfir.designate %[[VAL_28]] [%[[VAL_29]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_31:.*]] = arith.constant 3 : i32
! COARRAY:  %[[VAL_32:.*]] = fir.embox %[[VAL_30]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_33:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_or atom %[[VAL_32]][%[[VAL_29]]] value %[[VAL_31]] old %[[VAL_8]]#0 stat %[[VAL_33]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  %[[VAL_34:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
! COARRAY:  %[[VAL_35:.*]] = fir.box_addr %[[VAL_34]] : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
! COARRAY:  %[[VAL_36:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! COARRAY:  %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
! COARRAY:  %[[VAL_38:.*]] = hlfir.designate %[[VAL_35]] [%[[VAL_37]]] : (!fir.heap<i64>, i64) -> !fir.ref<i64>
! COARRAY:  %[[VAL_39:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_40:.*]] = fir.embox %[[VAL_38]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:  %[[VAL_41:.*]] = fir.absent !fir.ref<i32>
! COARRAY:  mif.atomic_xor atom %[[VAL_40]][%[[VAL_37]]] value %[[VAL_39]] old %[[VAL_8]]#0 stat %[[VAL_41]] : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
! COARRAY:  return
