// RUN: fir-opt --strict-fir-volatile-verifier %s --bufferize-hlfir --convert-hlfir-to-fir | FileCheck %s
func.func @_QQmain() attributes {fir.bindc_name = "p"} {
  %0 = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.volatile_cast %0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3:2 = hlfir.declare %2(%1) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %4 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %5 = fir.volatile_cast %4 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %6:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %7 = fir.volatile_cast %6#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
  fir.call @_QFPnot_declared_volatile_in_this_scope(%7) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
  %c1 = arith.constant 1 : index
  fir.do_loop %arg0 = %c1 to %c10 step %c1 {
    %22 = hlfir.designate %3#0 (%arg0)  : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.ref<i32, volatile>
    %23 = fir.volatile_cast %22 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
    fir.call @_QFPnot_declared_volatile_in_this_scope(%23) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
  }
  %c10_i32 = arith.constant 10 : i32
  %8 = fir.volatile_cast %3#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
  %9 = fir.convert %8 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %10:3 = hlfir.associate %c10_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  fir.call @_QFPdeclared_volatile_in_this_scope(%9, %10#0) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
  hlfir.end_associate %10#1, %10#2 : !fir.ref<i32>, i1
  %c6_i32 = arith.constant 6 : i32
  %11 = fir.address_of(@_QQclX28a011e93b63ba43ee03b06f1598b113) : !fir.ref<!fir.char<1,79>>
  %12 = fir.convert %11 : (!fir.ref<!fir.char<1,79>>) -> !fir.ref<i8>
  %c8_i32 = arith.constant 8 : i32
  %13 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %12, %c8_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %14 = fir.shape %c10 : (index) -> !fir.shape<1>
  %15 = fir.embox %3#0(%14) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
  %16 = fir.volatile_cast %15 : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
  %17 = fir.convert %16 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %18 = fir.call @_FortranAioOutputDescriptor(%13, %17) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
  %19 = fir.load %6#0 : !fir.ref<i32, volatile>
  %20 = fir.call @_FortranAioOutputInteger32(%13, %19) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
  %21 = fir.call @_FortranAioEndIoStatement(%13) fastmath<contract> : (!fir.ref<i8>) -> i32
  return
}
func.func private @_QFPnot_declared_volatile_in_this_scope(%arg0: !fir.ref<i32> {fir.bindc_name = "v"}) attributes {fir.host_symbol = @_QQmain, fir.proc_attrs = #fir.proc_attrs<elemental, pure>, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFFnot_declared_volatile_in_this_scopeEv"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %1#0 : i32, !fir.ref<i32>
  return
}
func.func private @_QFPdeclared_volatile_in_this_scope(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "v"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFdeclared_volatile_in_this_scopeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.convert %3 : (i64) -> index
  %c0 = arith.constant 0 : index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.volatile_cast %arg0 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>, volatile>
  %9:2 = hlfir.declare %8(%7) dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFdeclared_volatile_in_this_scopeEv"} : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>, volatile>, !fir.ref<!fir.array<?xi32>, volatile>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %9#0 : i32, !fir.box<!fir.array<?xi32>, volatile>
  return
}
fir.global internal @_QFEarr : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}

// CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref}
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
// CHECK:           %[[VAL_7:.*]] = fir.volatile_cast %[[VAL_6]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_9:.*]] = fir.volatile_cast %[[VAL_8]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:           fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_9]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_10]] to %[[VAL_2]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = fir.array_coor %[[VAL_5]](%[[VAL_3]]) %[[VAL_11]] : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, index) -> !fir.ref<i32, volatile>
// CHECK:             %[[VAL_13:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:             fir.call @_QFPnot_declared_volatile_in_this_scope(%[[VAL_13]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<i32>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = arith.constant false
// CHECK:           fir.call @_QFPdeclared_volatile_in_this_scope(%[[VAL_16]], %[[VAL_0]]) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
// CHECK:           %[[VAL_18:.*]] = arith.constant 6 : i32
// CHECK:           %[[VAL_19:.*]] = fir.address_of
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (
// CHECK:           %[[VAL_21:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_22:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_18]], %[[VAL_20]], %[[VAL_21]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
// CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_5]](%[[VAL_23]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_25:.*]] = fir.volatile_cast %[[VAL_24]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_27:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_22]], %[[VAL_26]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
// CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_29:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_22]], %[[VAL_28]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
// CHECK:           %[[VAL_30:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_22]]) fastmath<contract> : (!fir.ref<i8>) -> i32
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPnot_declared_volatile_in_this_scope(
// CHECK-SAME:                                                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "v"}) attributes {fir.host_symbol = @_QQmain, fir.proc_attrs = #fir.proc_attrs<elemental, pure>, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFFnot_declared_volatile_in_this_scopeEv"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPdeclared_volatile_in_this_scope(
// CHECK-SAME:                                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "v"},
// CHECK-SAME:                                                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?xi32>, volatile>
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFdeclared_volatile_in_this_scopeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>, volatile>
// CHECK:           %[[VAL_13:.*]] = fir.declare %[[VAL_12]](%[[VAL_11]]) dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFdeclared_volatile_in_this_scopeEv"} : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>, volatile>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_13]](%[[VAL_11]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>, volatile>
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_16:.*]] = fir.alloca i32
// CHECK:           fir.store %[[VAL_15]] to %[[VAL_16]] : !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_16]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_13]](%[[VAL_18]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>, volatile>
// CHECK:           fir.store %[[VAL_19]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xi32>, volatile>>
// CHECK:           %[[VAL_20:.*]] = fir.address_of(
// CHECK:           %[[VAL_21:.*]] = arith.constant
// CHECK:           %[[VAL_22:.*]] = arith.constant
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?xi32>, volatile>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_17]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_20]] : (
// CHECK:           fir.call @_FortranAAssign(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_22]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   fir.global internal @_QFEarr : !fir.array<10xi32> {
// CHECK:           %[[VAL_0:.*]] = fir.zero_bits !fir.array<10xi32>
// CHECK:           fir.has_value %[[VAL_0]] : !fir.array<10xi32>
// CHECK:         }
// CHECK:         func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}
