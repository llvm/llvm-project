! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

program atomic_add_array1d
  use, intrinsic :: iso_fortran_env, only: atomic_int_kind
  implicit none
  integer, parameter :: m = 5
  integer(atomic_int_kind) :: atom(m)[*], val
  integer :: me, n, i

  me = this_image(); n = num_images()
  atom = 0

  do i = 1, m
    call atomic_add(atom(i)[1], int(me*i, atomic_int_kind))
    call atomic_and(atom(i)   , int(me*i, atomic_int_kind))
    call atomic_or (atom(i)   , int(me*i, atomic_int_kind))
    call atomic_xor(atom(i)[1], int(me*i, atomic_int_kind))
  end do

end program

! COARRAY-LABEL: func.func @_QQmain()
! COARRAY:  %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY:  %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY:  %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY:  %[[VAL_3:.*]] = fir.address_of(@_QFEatom) : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:  %[[VAL_4:.*]] = arith.constant 1 : i64
! COARRAY:  %[[VAL_5:.*]] = arith.constant 0 : index
! COARRAY:  %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_5]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY:  fir.store %[[VAL_4]] to %[[VAL_6]] : !fir.ref<i64>
! COARRAY:  %[[VAL_7:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY:  %[[VAL_8:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY:  mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_7]] ucobounds %[[VAL_8]] uniq_name("_QFEatom") : (!fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! COARRAY:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFEatom")
! COARRAY:  %[[VAL_10:.*]] = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
! COARRAY:  %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter>
! COARRAY:  %[[VAL_12:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFEi"}>
! COARRAY:  %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] uniq_name("_QFEi")
! COARRAY:  %[[VAL_14:.*]] = fir.address_of(@_QFECm) : !fir.ref<i32>
! COARRAY:  %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] uniq_name("_QFECm") fortran_attrs<parameter>
! COARRAY:  %[[VAL_16:.*]] = fir.alloca i32 <{bindc_name = "me", uniq_name = "_QFEme"}>
! COARRAY:  %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] uniq_name("_QFEme")
! COARRAY:  %[[VAL_18:.*]] = fir.alloca i32 <{bindc_name = "n", uniq_name = "_QFEn"}>
! COARRAY:  %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]] uniq_name("_QFEn")
! COARRAY:  %[[VAL_20:.*]] = fir.alloca i64 <{bindc_name = "val", uniq_name = "_QFEval"}>
! COARRAY:  %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_20]] uniq_name("_QFEval")
! COARRAY:  %[[VAL_22:.*]] = mif.this_image : () -> i32
! COARRAY:  hlfir.assign %[[VAL_22]] to %[[VAL_17]]#0 : i32, !fir.ref<i32>
! COARRAY:  %[[VAL_23:.*]] = mif.num_images : () -> i32
! COARRAY:  hlfir.assign %[[VAL_23]] to %[[VAL_19]]#0 : i32, !fir.ref<i32>
! COARRAY:  %[[VAL_24:.*]] = arith.constant 0 : i64
! COARRAY:  %[[VAL_25:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:  hlfir.assign %[[VAL_24]] to %[[VAL_25]] : i64, !fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>
! COARRAY:  %[[VAL_26:.*]] = arith.constant 1 : i32
! COARRAY:  %[[VAL_27:.*]] = arith.constant 5 : i32
! COARRAY:  %[[VAL_28:.*]] = arith.constant 1 : i32
! COARRAY:  fir.do_loop %[[VAL_29:.*]] = %[[VAL_26]] to %[[VAL_27]] step %[[VAL_28]] : i32 {
! COARRAY:    fir.store %[[VAL_29]] to %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_30:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:    %[[VAL_31:.*]] = arith.constant 1 : i64
! COARRAY:    %[[VAL_32:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! COARRAY:    %[[VAL_34:.*]] = hlfir.designate %[[VAL_30]] (%[[VAL_33]])[%[[VAL_31]]] : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64, i64) -> !fir.ref<i64>
! COARRAY:    %[[VAL_35:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_36:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_37:.*]] = arith.muli %[[VAL_35]], %[[VAL_36]] : i32
! COARRAY:    %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
! COARRAY:    %[[VAL_39:.*]] = fir.embox %[[VAL_34]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:    %[[VAL_40:.*]] = fir.absent !fir.ref<i32>
! COARRAY:    mif.atomic_add atom %[[VAL_39]][%[[VAL_31]]] value %[[VAL_38]] stat %[[VAL_40]] : (!fir.box<i64, corank:1>, i64, i64, !fir.ref<i32>) -> ()
! COARRAY:    %[[VAL_41:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:    %[[VAL_42:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> i64
! COARRAY:    %[[VAL_44:.*]] = hlfir.designate %[[VAL_41]] (%[[VAL_43]]) : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64) -> !fir.ref<i64>
! COARRAY:    %[[VAL_45:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_46:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_47:.*]] = arith.muli %[[VAL_45]], %[[VAL_46]] : i32
! COARRAY:    %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
! COARRAY:    %[[VAL_49:.*]] = fir.embox %[[VAL_44]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:    %[[VAL_50:.*]] = fir.absent !fir.ref<i32>
! COARRAY:    mif.atomic_and atom %[[VAL_49]] value %[[VAL_48]] stat %[[VAL_50]] : (!fir.box<i64, corank:1>, i64, !fir.ref<i32>) -> ()
! COARRAY:    %[[VAL_51:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:    %[[VAL_52:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> i64
! COARRAY:    %[[VAL_54:.*]] = hlfir.designate %[[VAL_51]] (%[[VAL_53]]) : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64) -> !fir.ref<i64>
! COARRAY:    %[[VAL_55:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_56:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_57:.*]] = arith.muli %[[VAL_55]], %[[VAL_56]] : i32
! COARRAY:    %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i32) -> i64
! COARRAY:    %[[VAL_59:.*]] = fir.embox %[[VAL_54]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:    %[[VAL_60:.*]] = fir.absent !fir.ref<i32>
! COARRAY:    mif.atomic_or atom %[[VAL_59]] value %[[VAL_58]] stat %[[VAL_60]] : (!fir.box<i64, corank:1>, i64, !fir.ref<i32>) -> ()
! COARRAY:    %[[VAL_61:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
! COARRAY:    %[[VAL_62:.*]] = arith.constant 1 : i64
! COARRAY:    %[[VAL_63:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
! COARRAY:    %[[VAL_65:.*]] = hlfir.designate %[[VAL_61]] (%[[VAL_64]])[%[[VAL_62]]] : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64, i64) -> !fir.ref<i64>
! COARRAY:    %[[VAL_66:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_67:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:    %[[VAL_68:.*]] = arith.muli %[[VAL_66]], %[[VAL_67]] : i32
! COARRAY:    %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (i32) -> i64
! COARRAY:    %[[VAL_70:.*]] = fir.embox %[[VAL_65]] : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
! COARRAY:    %[[VAL_71:.*]] = fir.absent !fir.ref<i32>
! COARRAY:    mif.atomic_xor atom %[[VAL_70]][%[[VAL_62]]] value %[[VAL_69]] stat %[[VAL_71]] : (!fir.box<i64, corank:1>, i64, i64, !fir.ref<i32>) -> ()
! COARRAY:  }
! COARRAY:  %[[VAL_72:.*]] = fir.convert %[[VAL_26]] : (i32) -> index
! COARRAY:  %[[VAL_73:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! COARRAY:  %[[VAL_74:.*]] = fir.convert %[[VAL_28]] : (i32) -> index
! COARRAY:  %[[VAL_75:.*]] = arith.constant 0 : index
! COARRAY:  %[[VAL_76:.*]] = arith.subi %[[VAL_73]], %[[VAL_72]] : index
! COARRAY:  %[[VAL_77:.*]] = arith.addi %[[VAL_76]], %[[VAL_74]] : index
! COARRAY:  %[[VAL_78:.*]] = arith.divsi %[[VAL_77]], %[[VAL_74]] : index
! COARRAY:  %[[VAL_79:.*]] = arith.cmpi slt, %[[VAL_78]], %[[VAL_75]] : index
! COARRAY:  %[[VAL_80:.*]] = arith.select %[[VAL_79]], %[[VAL_75]], %[[VAL_78]] : index
! COARRAY:  %[[VAL_81:.*]] = arith.muli %[[VAL_80]], %[[VAL_74]] : index
! COARRAY:  %[[VAL_82:.*]] = arith.addi %[[VAL_72]], %[[VAL_81]] : index
! COARRAY:  %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (index) -> i32
! COARRAY:  fir.store %[[VAL_83]] to %[[VAL_13]]#0 : !fir.ref<i32>
! COARRAY:  return
