// RUN: fir-opt --function-attr="set-noalias=true" %s | FileCheck %s

// Test the annotation of function arguments with llvm.noalias.

// Test !fir.ref arguments.
// CHECK-LABEL:   func.func private @test_ref(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {llvm.noalias}) {
func.func private @test_ref(%arg0: !fir.ref<i32>) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_target(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.target}) {
func.func private @test_ref_target(%arg0: !fir.ref<i32> {fir.target}) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_volatile(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.volatile}) {
func.func private @test_ref_volatile(%arg0: !fir.ref<i32> {fir.volatile}) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_asynchronous(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.asynchronous}) {
func.func private @test_ref_asynchronous(%arg0: !fir.ref<i32> {fir.asynchronous}) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_box(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<i32>> {llvm.noalias}) {
// Test !fir.ref<!fir.box<>> arguments:
func.func private @test_ref_box(%arg0: !fir.ref<!fir.box<i32>>) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_box_target(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<i32>> {fir.target}) {
func.func private @test_ref_box_target(%arg0: !fir.ref<!fir.box<i32>> {fir.target}) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_box_volatile(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<i32>> {fir.volatile}) {
func.func private @test_ref_box_volatile(%arg0: !fir.ref<!fir.box<i32>> {fir.volatile}) {
  return
}

// CHECK-LABEL:   func.func private @test_ref_box_asynchronous(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<i32>> {fir.asynchronous}) {
func.func private @test_ref_box_asynchronous(%arg0: !fir.ref<!fir.box<i32>> {fir.asynchronous}) {
  return
}

// Test POINTER arguments.
// CHECK-LABEL:   func.func private @test_ref_box_ptr(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>) {
func.func private @test_ref_box_ptr(%arg0: !fir.ref<!fir.box<!fir.ptr<i32>>>) {
  return
}

// Test ALLOCATABLE arguments.
// CHECK-LABEL:   func.func private @test_ref_box_heap(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {llvm.noalias}) {
func.func private @test_ref_box_heap(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>>) {
  return
}

// BIND(C) functions are not annotated.
// CHECK-LABEL:   func.func private @test_ref_bindc(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32>)
func.func private @test_ref_bindc(%arg0: !fir.ref<i32>) attributes {fir.bindc_name = "test_ref_bindc", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  return
}

// Test function declaration from a module.
// CHECK-LABEL:   func.func private @_QMtest_modPcheck_module(
// CHECK-SAME:      !fir.ref<i32> {llvm.noalias})
func.func private @_QMtest_modPcheck_module(!fir.ref<i32>)

// Test !fir.box arguments:
// CHECK-LABEL:   func.func private @test_box(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<i32> {llvm.noalias}) {
func.func private @test_box(%arg0: !fir.box<i32>) {
  return
}

// CHECK-LABEL:   func.func private @test_box_target(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<i32> {fir.target, llvm.noalias}) {
func.func private @test_box_target(%arg0: !fir.box<i32> {fir.target}) {
  return
}

// CHECK-LABEL:   func.func private @test_box_volatile(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<i32> {fir.volatile, llvm.noalias}) {
func.func private @test_box_volatile(%arg0: !fir.box<i32> {fir.volatile}) {
  return
}

// CHECK-LABEL:   func.func private @test_box_asynchronous(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<i32> {fir.asynchronous, llvm.noalias}) {
func.func private @test_box_asynchronous(%arg0: !fir.box<i32> {fir.asynchronous}) {
  return
}

// !fir.boxchar<> is lowered before FunctionAttrPass, but let's
// make sure we do not annotate it.
// CHECK-LABEL:   func.func private @test_boxchar(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1>) {
func.func private @test_boxchar(%arg0: !fir.boxchar<1>) {
  return
}

