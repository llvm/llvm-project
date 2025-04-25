// RUN: fir-opt %s --bufferize-hlfir --convert-hlfir-to-fir | FileCheck %s
func.func @_QQmain() attributes {fir.bindc_name = "p"} {
  %0 = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.volatile_cast %0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3:2 = hlfir.declare %2(%1) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %4 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %5 = fir.volatile_cast %4 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %6:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %7 = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %8 = fir.volatile_cast %7 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %9:2 = hlfir.declare %8 {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
  %10 = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
  %c10_0 = arith.constant 10 : index
  %11 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %12 = fir.volatile_cast %10 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %13:2 = hlfir.declare %12(%11) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %14 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %15 = fir.embox %13#0(%14) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
  fir.store %15 to %9#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %c0_i32 = arith.constant 0 : i32
  hlfir.assign %c0_i32 to %6#0 : i32, !fir.ref<i32, volatile>
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %3#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
  %16 = fir.volatile_cast %3#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
  fir.call @_QFPd(%16) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
  %17 = fir.embox %3#0(%1) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
  %18 = fir.volatile_cast %17 : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
  %19 = fir.convert %18 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
  fir.call @_QFPe(%19) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
  %20 = fir.volatile_cast %3#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
  fir.call @_QFPf(%20) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
  %21 = fir.convert %9#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
  fir.call @_QFPg(%21) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
  return
}
func.func private @_QFPd(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.volatile_cast %arg0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3:2 = hlfir.declare %2(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFdEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  return
}
func.func private @_QFPe(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFeEarr"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  return
}
func.func private @_QFPf(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.volatile_cast %arg0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3:2 = hlfir.declare %2(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFfEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  return
}
func.func private @_QFPg(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.volatile_cast %arg0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %2:2 = hlfir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFFgEarr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
  return
}

// CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
// CHECK:           %[[VAL_7:.*]] = fir.volatile_cast %[[VAL_6]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_10:.*]] = fir.volatile_cast %[[VAL_9]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_11:.*]] = fir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_13:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_16:.*]] = fir.declare %[[VAL_15]](%[[VAL_14]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : i32
// CHECK:           fir.store %[[VAL_19]] to %[[VAL_8]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_21:.*]] = fir.alloca i32
// CHECK:           fir.store %[[VAL_20]] to %[[VAL_21]] : !fir.ref<i32>
// CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_21]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_5]](%[[VAL_23]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           fir.store %[[VAL_24]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<10xi32>, volatile>>
// CHECK:           %[[VAL_25:.*]] = fir.address_of
// CHECK:           %[[VAL_26:.*]] = arith.constant
// CHECK:           %[[VAL_27:.*]] = arith.constant
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<10xi32>, volatile>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_22]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_25]]
// CHECK:           fir.call @_FortranAAssign(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_27]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_31:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
// CHECK:           fir.call @_QFPd(%[[VAL_31]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
// CHECK:           %[[VAL_32:.*]] = fir.embox %[[VAL_5]](%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_33:.*]] = fir.volatile_cast %[[VAL_32]] : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           fir.call @_QFPe(%[[VAL_34]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
// CHECK:           %[[VAL_35:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<10xi32>>
// CHECK:           fir.call @_QFPf(%[[VAL_35]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>
// CHECK:           fir.call @_QFPg(%[[VAL_36]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPd(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]](%[[VAL_3]]) dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFdEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPe(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFeEarr"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPf(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]](%[[VAL_3]]) dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFfEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPg(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile> {fir.bindc_name = "arr"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_2]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFFgEarr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           return
// CHECK:         }
