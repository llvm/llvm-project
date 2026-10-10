// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" --mlir-print-debuginfo %s | FileCheck %s

// A bound or length that lowering stored to a slot marked with
// fir.debug_bound_slot is described by the slot, with a declaration that
// becomes a dbg.declare, and does not need a fake use.

#loc1 = loc("test.f90":1:1)
#loc3 = loc("test.f90":3:14)
#loc5 = loc("test.f90":5:1)
#loc = loc("test.f90":0:0)

module {
  func.func private @foo()
  func.func @test_dynamic_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc(#loc1), %arg1: index loc(#loc1), %arg2: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_dynamic_array"} {
    %0 = fir.alloca index {fir.debug_bound_slot} loc(#loc3)
    %1 = fir.alloca index {fir.debug_bound_slot} loc(#loc3)
    %2 = fir.undefined !fir.dscope loc(#loc1)
    fir.store %arg2 to %1 : !fir.ref<index> loc(#loc3)
    fir.store %arg1 to %0 : !fir.ref<index> loc(#loc3)
    %3 = fircg.ext_declare %arg0(%arg1) origin %arg2 dummy_scope %2 arg 1 uniq_name("_QFtest_dynamic_arrayEarr") : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_dynamic_array(
// CHECK:         %[[COUNT_SLOT:.*]] = fir.alloca index
// CHECK:         %[[LB_SLOT:.*]] = fir.alloca index
// CHECK-NOT:     llvm.intr.dbg.value
// CHECK:         fircg.ext_declare %[[LB_SLOT]] uniq_name("._QFtest_dynamic_arrayEarr2") : (!fir.ref<index>) -> !fir.ref<index>
// CHECK:         fircg.ext_declare %[[COUNT_SLOT]] uniq_name("._QFtest_dynamic_arrayEarr1") : (!fir.ref<index>) -> !fir.ref<index>
// CHECK:         fircg.ext_declare %arg0(%arg1) origin %arg2
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    return

  func.func @test_char(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str"} loc(#loc1), %arg1: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_char"} {
    %0 = fir.alloca index {fir.debug_bound_slot} loc(#loc3)
    %1 = fir.undefined !fir.dscope loc(#loc1)
    fir.store %arg1 to %0 : !fir.ref<index> loc(#loc3)
    %2 = fircg.ext_declare %arg0 typeparams %arg1 dummy_scope %1 arg 1 uniq_name("_QFtest_charEstr") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_char(
// CHECK:         %[[LEN_SLOT:.*]] = fir.alloca index
// CHECK-NOT:     llvm.intr.dbg.value
// CHECK:         fircg.ext_declare %[[LEN_SLOT]] uniq_name("._QFtest_charEstr1") : (!fir.ref<index>) -> !fir.ref<index>
// CHECK:         fircg.ext_declare %arg0 typeparams %arg1
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    return

  // An alloca of the same shape that lowering did not mark is left alone.
  func.func @test_unmarked_slot(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc(#loc1), %arg1: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_unmarked_slot"} {
    %0 = fir.alloca index loc(#loc3)
    %1 = fir.undefined !fir.dscope loc(#loc1)
    fir.store %arg1 to %0 : !fir.ref<index> loc(#loc3)
    %2 = fircg.ext_declare %arg0(%arg1) dummy_scope %1 arg 1 uniq_name("_QFtest_unmarked_slotEarr") : (!fir.ref<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_unmarked_slot(
// CHECK:         %[[COUNT:.*]] = fir.convert %arg1 : (index) -> i64
// CHECK:         llvm.intr.dbg.value #{{.*}} = %[[COUNT]] : i64
// CHECK:         fircg.ext_declare %arg0(%arg1)
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    fir.fake_use %arg1
// CHECK-NEXT:    return

  // A second store means the slot may not hold the count everywhere.
  func.func @test_slot_overwritten(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc(#loc1), %arg1: index loc(#loc1), %arg2: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_slot_overwritten"} {
    %0 = fir.alloca index {fir.debug_bound_slot} loc(#loc3)
    %1 = fir.undefined !fir.dscope loc(#loc1)
    fir.store %arg1 to %0 : !fir.ref<index> loc(#loc3)
    %2 = fircg.ext_declare %arg0(%arg1) dummy_scope %1 arg 1 uniq_name("_QFtest_slot_overwrittenEarr") : (!fir.ref<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.store %arg2 to %0 : !fir.ref<index> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_slot_overwritten(
// CHECK:         %[[COUNT2:.*]] = fir.convert %arg1 : (index) -> i64
// CHECK:         llvm.intr.dbg.value #{{.*}} = %[[COUNT2]] : i64
// CHECK:         fircg.ext_declare %arg0(%arg1)
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    fir.fake_use %arg1
// CHECK-NEXT:    return
} loc(#loc)
