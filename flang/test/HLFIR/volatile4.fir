// RUN: fir-opt %s --bufferize-hlfir --convert-hlfir-to-fir | FileCheck %s
func.func @_QQmain() attributes {fir.bindc_name = "p"} {
  %0 = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.volatile_cast %0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3:2 = hlfir.declare %2(%1) {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %4 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %5 = fir.volatile_cast %4 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %6:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %7 = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %8 = fir.volatile_cast %7 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %9:2 = hlfir.declare %8 {fortran_attrs = #fir.var_attrs<pointer, volatile, internal_assoc>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
  %10 = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
  %c10_0 = arith.constant 10 : index
  %11 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %12 = fir.volatile_cast %10 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %13:2 = hlfir.declare %12(%11) {fortran_attrs = #fir.var_attrs<target, volatile, internal_assoc>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %14 = fir.alloca tuple<!fir.ref<i32>>
  %c0_i32 = arith.constant 0 : i32
  %15 = fir.coordinate_of %14, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  %16 = fir.volatile_cast %6#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
  fir.store %16 to %15 : !fir.llvm_ptr<!fir.ref<i32>>
  %17 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %18 = fir.embox %13#0(%17) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
  fir.store %18 to %9#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %c0_i32_1 = arith.constant 0 : i32
  hlfir.assign %c0_i32_1 to %6#0 : i32, !fir.ref<i32, volatile>
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %3#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
  fir.call @_QFPhost_assoc(%14) fastmath<contract> : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  return
}
func.func private @_QFPhost_assoc(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3 = fir.volatile_cast %1 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %4:2 = hlfir.declare %3(%2) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %5 = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %6 = fir.volatile_cast %5 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %7:2 = hlfir.declare %6 {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
  %8 = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
  %c10_0 = arith.constant 10 : index
  %9 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %10 = fir.volatile_cast %8 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %11:2 = hlfir.declare %10(%9) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
  %c0_i32 = arith.constant 0 : i32
  %12 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  %13 = fir.load %12 : !fir.llvm_ptr<!fir.ref<i32>>
  %14 = fir.volatile_cast %13 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %15:2 = hlfir.declare %14 {fortran_attrs = #fir.var_attrs<volatile, host_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %16 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %17 = fir.embox %11#0(%16) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
  fir.store %17 to %7#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
  %c0_i32_1 = arith.constant 0 : i32
  hlfir.assign %c0_i32_1 to %15#0 : i32, !fir.ref<i32, volatile>
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %4#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
  return
}
// CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]](%[[VAL_3]]) {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
// CHECK:           %[[VAL_7:.*]] = fir.volatile_cast %[[VAL_6]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_10:.*]] = fir.volatile_cast %[[VAL_9]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_11:.*]] = fir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<pointer, volatile, internal_assoc>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_13:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]] = fir.volatile_cast %[[VAL_12]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_16:.*]] = fir.declare %[[VAL_15]](%[[VAL_14]]) {fortran_attrs = #fir.var_attrs<target, volatile, internal_assoc>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_17:.*]] = fir.alloca tuple<!fir.ref<i32>>
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_18]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:           %[[VAL_20:.*]] = fir.volatile_cast %[[VAL_8]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:           fir.store %[[VAL_20]] to %[[VAL_19]] : !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:           %[[VAL_21:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_16]](%[[VAL_21]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
// CHECK:           fir.store %[[VAL_22]] to %[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:           fir.store %[[VAL_23]] to %[[VAL_8]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_25:.*]] = fir.alloca i32
// CHECK:           fir.store %[[VAL_24]] to %[[VAL_25]] : !fir.ref<i32>
// CHECK:           %[[VAL_26:.*]] = fir.embox %[[VAL_25]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_27:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_5]](%[[VAL_27]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           fir.store %[[VAL_28]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<10xi32>, volatile>>
// CHECK:           %[[VAL_29:.*]] = fir.address_of
// CHECK:           %[[VAL_30:.*]] = arith.constant
// CHECK:           %[[VAL_31:.*]] = arith.constant
// CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<10xi32>, volatile>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_26]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_29]]
// CHECK:           fir.call @_FortranAAssign(%[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_31]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           fir.call @_QFPhost_assoc(%[[VAL_17]]) fastmath<contract> : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPhost_assoc(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_3]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_6]](%[[VAL_5]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_9:.*]] = fir.volatile_cast %[[VAL_8]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
// CHECK:           %[[VAL_12:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]] = fir.volatile_cast %[[VAL_11]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_15:.*]] = fir.declare %[[VAL_14]](%[[VAL_13]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>, volatile>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_16]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:           %[[VAL_19:.*]] = fir.volatile_cast %[[VAL_18]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_20:.*]] = fir.declare %[[VAL_19]] {fortran_attrs = #fir.var_attrs<volatile, host_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_21:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_15]](%[[VAL_21]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
// CHECK:           fir.store %[[VAL_22]] to %[[VAL_10]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:           fir.store %[[VAL_23]] to %[[VAL_20]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_25:.*]] = fir.alloca i32
// CHECK:           fir.store %[[VAL_24]] to %[[VAL_25]] : !fir.ref<i32>
// CHECK:           %[[VAL_26:.*]] = fir.embox %[[VAL_25]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_27:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_7]](%[[VAL_27]]) : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>, volatile>
// CHECK:           fir.store %[[VAL_28]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<10xi32>, volatile>>
// CHECK:           %[[VAL_29:.*]] = fir.address_of
// CHECK:           %[[VAL_30:.*]] = arith.constant
// CHECK:           %[[VAL_31:.*]] = arith.constant
// CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.array<10xi32>, volatile>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_26]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_29]]
// CHECK:           fir.call @_FortranAAssign(%[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_31]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}

