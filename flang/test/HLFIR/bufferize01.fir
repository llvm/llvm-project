// RUN: fir-opt --bufferize-hlfir --split-input-file %s | FileCheck %s

// -----

// Bufferization for hlfir.apply and hlfir.no_reassoc must establish
// the <storage, mustFree> tuple properly, so that the users have
// access to both components.

// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 80 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>> {bindc_name = ".tmp.arrayctor"}
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "w", uniq_name = "_QFtest1Ew"}
// CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
// CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_8]] typeparams %[[VAL_4]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK:           fir.store %[[VAL_9]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_7]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest1Ew"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
// CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<1x!fir.char<1,?>>>
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_11]](%[[VAL_12]]) typeparams %[[VAL_4]] : (!fir.heap<!fir.array<1x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
// CHECK:           %[[VAL_15:.*]] = fir.address_of(@_QQclXce30ef70ff16a711a97719fb946c0b3d) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAInitArrayConstructorVector(%[[VAL_14]], %[[VAL_16]], %[[VAL_2]], %[[VAL_1]], %[[VAL_17]], %[[VAL_0]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_22:.*]] = fir.box_elesize %[[VAL_21]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
// CHECK:           %[[VAL_23:.*]] = fir.emboxchar %[[VAL_20]], %[[VAL_22]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_25:.*]] = fir.box_elesize %[[VAL_24]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
// CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_4]] : index
// CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_4]] : index
// CHECK:           %[[VAL_30:.*]] = hlfir.designate %[[VAL_23]]  substr %[[VAL_3]], %[[VAL_27]]  typeparams %[[VAL_29]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_31:.*]]:2 = fir.unboxchar %[[VAL_30]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_32:.*]] = fir.embox %[[VAL_31]]#0 typeparams %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
// CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranAPushArrayConstructorValue(%[[VAL_14]], %[[VAL_33]]) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>
// CHECK:           %[[VAL_36:.*]] = fir.undefined tuple<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1>
// CHECK:           %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_2]], [1 : index] : (tuple<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1>, i1) -> tuple<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1>
// CHECK:           %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_35]], [0 : index] : (tuple<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1>, !fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> tuple<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1>
// CHECK:           %[[VAL_39:.*]] = fir.box_elesize %[[VAL_35]] : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> index
// CHECK:           %[[VAL_40:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_41:.*]] = fir.allocmem !fir.array<1x!fir.char<1,?>>(%[[VAL_39]] : index) {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_42:.*]]:2 = hlfir.declare %[[VAL_41]](%[[VAL_40]]) typeparams %[[VAL_39]] {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<1x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<1x!fir.char<1,?>>>, !fir.heap<!fir.array<1x!fir.char<1,?>>>)
// CHECK:           %[[VAL_43:.*]] = arith.constant true
// CHECK:           %[[VAL_44:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_45:.*]] = %[[VAL_44]] to %[[VAL_3]] step %[[VAL_44]] {
// CHECK:             %[[VAL_46:.*]] = fir.box_elesize %[[VAL_35]] : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> index
// CHECK:             %[[VAL_47:.*]] = hlfir.designate %[[VAL_35]] (%[[VAL_45]])  typeparams %[[VAL_46]] : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, index, index) -> !fir.boxchar<1>
// CHECK:             %[[VAL_48:.*]] = arith.constant false
// CHECK:             %[[VAL_49:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_50:.*]] = fir.insert_value %[[VAL_49]], %[[VAL_48]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_51:.*]] = fir.insert_value %[[VAL_50]], %[[VAL_47]], [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_52:.*]] = hlfir.no_reassoc %[[VAL_47]] : !fir.boxchar<1>
// CHECK:             %[[VAL_53:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_54:.*]] = fir.insert_value %[[VAL_53]], %[[VAL_48]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_55:.*]] = fir.insert_value %[[VAL_54]], %[[VAL_52]], [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_56:.*]] = hlfir.designate %[[VAL_42]]#0 (%[[VAL_45]])  typeparams %[[VAL_39]] : (!fir.box<!fir.array<1x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:             hlfir.assign %[[VAL_52]] to %[[VAL_56]] temporary_lhs : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = fir.undefined tuple<!fir.box<!fir.array<1x!fir.char<1,?>>>, i1>
// CHECK:           %[[VAL_58:.*]] = fir.insert_value %[[VAL_57]], %[[VAL_43]], [1 : index] : (tuple<!fir.box<!fir.array<1x!fir.char<1,?>>>, i1>, i1) -> tuple<!fir.box<!fir.array<1x!fir.char<1,?>>>, i1>
// CHECK:           %[[VAL_59:.*]] = fir.insert_value %[[VAL_58]], %[[VAL_42]]#0, [0 : index] : (tuple<!fir.box<!fir.array<1x!fir.char<1,?>>>, i1>, !fir.box<!fir.array<1x!fir.char<1,?>>>) -> tuple<!fir.box<!fir.array<1x!fir.char<1,?>>>, i1>
// CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_42]]#1 : (!fir.heap<!fir.array<1x!fir.char<1,?>>>) -> !fir.ref<!fir.array<1x!fir.char<1,?>>>
// CHECK:           %[[VAL_61:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_62:.*]]:2 = hlfir.declare %[[VAL_60]](%[[VAL_61]]) typeparams %[[VAL_39]] {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.array<1x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<1x!fir.char<1,?>>>, !fir.ref<!fir.array<1x!fir.char<1,?>>>)
// CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_60]] : (!fir.ref<!fir.array<1x!fir.char<1,?>>>) -> !fir.heap<!fir.array<1x!fir.char<1,?>>>
// CHECK:           fir.freemem %[[VAL_63]] : !fir.heap<!fir.array<1x!fir.char<1,?>>>
// CHECK:           %[[VAL_64:.*]] = fir.box_addr %[[VAL_35]] : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<1x!fir.char<1,?>>>
// CHECK:           fir.freemem %[[VAL_64]] : !fir.heap<!fir.array<1x!fir.char<1,?>>>
// CHECK:           return
// CHECK:         }
func.func @_QPtest1() {
  %c1_i32 = arith.constant 1 : i32
  %c80_i32 = arith.constant 80 : i32
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>> {bindc_name = ".tmp.arrayctor"}
  %2 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "w", uniq_name = "_QFtest1Ew"}
  %3 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %4 = fir.embox %3 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %4 to %2 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %5:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest1Ew"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
  %6 = fir.zero_bits !fir.heap<!fir.array<1x!fir.char<1,?>>>
  %7 = fir.shape %c1 : (index) -> !fir.shape<1>
  %8 = fir.embox %6(%7) typeparams %c0 : (!fir.heap<!fir.array<1x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>
  fir.store %8 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>
  %9 = fir.convert %0 : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
  %10 = fir.address_of(@_QQclXce30ef70ff16a711a97719fb946c0b3d) : !fir.ref<!fir.char<1,1>>
  %11 = fir.convert %1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %10 : (!fir.ref<!fir.char<1,1>>) -> !fir.ref<i8>
  fir.call @_FortranAInitArrayConstructorVector(%9, %11, %true, %c80_i32, %12, %c1_i32) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> ()
  %14 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %16 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %17 = fir.box_elesize %16 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %18 = fir.emboxchar %15, %17 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  %19 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %20 = fir.box_elesize %19 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %21 = fir.convert %20 : (index) -> i64
  %22 = fir.convert %21 : (i64) -> index
  %23 = arith.cmpi sgt, %22, %c0 : index
  %24 = arith.select %23, %22, %c0 : index
  %25 = hlfir.designate %18  substr %c1, %22  typeparams %24 : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
  %26:2 = fir.unboxchar %25 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %27 = fir.embox %26#0 typeparams %24 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  %28 = fir.convert %27 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  fir.call @_FortranAPushArrayConstructorValue(%9, %28) fastmath<contract> : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
  %30 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>>
  %31 = hlfir.as_expr %30 move %true : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>, i1) -> !hlfir.expr<1x!fir.char<1,?>>
  %32 = fir.box_elesize %30 : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> index
  %33 = fir.shape %c1 : (index) -> !fir.shape<1>
  %34 = hlfir.elemental %33 typeparams %32 : (!fir.shape<1>, index) -> !hlfir.expr<1x!fir.char<1,?>> {
  ^bb0(%arg0: index):
    %38 = fir.box_elesize %30 : (!fir.box<!fir.heap<!fir.array<1x!fir.char<1,?>>>>) -> index
    %39 = hlfir.apply %31, %arg0 typeparams %38 : (!hlfir.expr<1x!fir.char<1,?>>, index, index) -> !hlfir.expr<!fir.char<1,?>>
    %40 = hlfir.no_reassoc %39 : !hlfir.expr<!fir.char<1,?>>
    hlfir.yield_element %40 : !hlfir.expr<!fir.char<1,?>>
  }
  %35:3 = hlfir.associate %34(%33) typeparams %32 {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<1x!fir.char<1,?>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<1x!fir.char<1,?>>>, !fir.ref<!fir.array<1x!fir.char<1,?>>>, i1)
  %36 = fir.shape %c1 : (index) -> !fir.shape<1>
  %37:2 = hlfir.declare %35#1(%36) typeparams %32 {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.array<1x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<1x!fir.char<1,?>>>, !fir.ref<!fir.array<1x!fir.char<1,?>>>)
  hlfir.end_associate %35#1, %35#2 : !fir.ref<!fir.array<1x!fir.char<1,?>>>, i1
  hlfir.destroy %34 : !hlfir.expr<1x!fir.char<1,?>>
  hlfir.destroy %31 : !hlfir.expr<1x!fir.char<1,?>>
  return
}
func.func private @_FortranAInitArrayConstructorVector(!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> () attributes {fir.runtime}
fir.global linkonce @_QQclXce30ef70ff16a711a97719fb946c0b3d constant : !fir.char<1,1> {
  %0 = fir.string_lit "\00"(1) : !fir.char<1,1>
  fir.has_value %0 : !fir.char<1,1>
}
func.func private @_FortranAPushArrayConstructorValue(!fir.llvm_ptr<i8>, !fir.box<none>) -> () attributes {fir.runtime}

// -----

// Test that only a single freemem is generated.

func.func @double_free(%arg0: !fir.boxchar<1>) {
  %c5 = arith.constant 5 : index
  %true = arith.constant true
  %0 = hlfir.as_expr %arg0 move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  %1 = hlfir.no_reassoc %0 : !hlfir.expr<!fir.char<1,?>>
  %2:3 = hlfir.associate %1 typeparams %c5 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @noop(%2#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %2#1, %2#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
} 
func.func private @noop(!fir.boxchar<1>)

// CHECK-LABEL: func.func @double_free(
// CHECK-SAME:  %[[ARG0:.*]]: !fir.boxchar<1>) {
// CHECK: %[[NO_REASSOC:.*]] = hlfir.no_reassoc %[[ARG0]] : !fir.boxchar<1>
// CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[NO_REASSOC]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
// CHECK: fir.call @noop
// CHECK: %[[CONV:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
// CHECK: fir.freemem %[[CONV]] : !fir.heap<!fir.char<1,?>>
// CHECK-NOT: fir.freemem


