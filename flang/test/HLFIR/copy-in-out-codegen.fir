// Test hlfir.copy_in and hlfir.copy_out operation codegen

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_copy_in(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.array<?xf64>>>) {
  %0:3 = hlfir.copy_in %box to %temp : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.array<?xf64>>>) -> (!fir.box<!fir.array<?xf64>>, i1, i1)
  return
}
// CHECK-LABEL: func.func @test_copy_in(
// CHECK:         fir.allocmem !fir.array<?xf64>
// CHECK:         fir.embox
// CHECK:         fir.call @_FortranAShallowCopyDirect
// CHECK:         fir.store %{{.*}} to %{{.*}} : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK-NOT:     fir.call @_FortranACopyInAssign

func.func @test_copy_in_poly_pair(%poly : !fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>, %temp: !fir.ref<!fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>>) {
  %0:3 = hlfir.copy_in %poly to %temp : (!fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>, !fir.ref<!fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>>) -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>, i1, i1)
  hlfir.copy_out %temp, %0#1, %0#2 : (!fir.ref<!fir.class<!fir.array<?x!fir.type<test_copy_in_poly_pairTt1{i:i32}>>>>, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @test_copy_in_poly_pair(
// CHECK:         fir.call @_FortranACopyInAssign
// CHECK:         fir.freemem {{.*}} : !fir.heap<i8>

func.func @test_optional_copy_pair(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.array<?xf64>>>, %is_present: i1) {
  %0:3 = hlfir.copy_in %box to %temp handle_optional %is_present : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.array<?xf64>>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1, i1)
  hlfir.copy_out %temp, %0#1, %0#2 to %box : (!fir.ref<!fir.box<!fir.array<?xf64>>>, i1, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL: func.func @test_optional_copy_pair(
// CHECK:         fir.if %{{.*}} -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:           fir.if %{{.*}} -> (!fir.box<!fir.array<?xf64>>) {
// CHECK:           } else {
// CHECK:             fir.allocmem !fir.array<?xf64>
// CHECK:             fir.embox
// CHECK:             fir.call @_FortranAShallowCopyDirect
// CHECK:           }
// CHECK:         }
// CHECK:         fir.if %{{.*}} {
// CHECK:           fir.call @_FortranACopyOutAssignDirect
// CHECK:         fir.if %{{.*}} {
// CHECK:           fir.freemem
// CHECK-NOT:     fir.call @_FortranACopyInAssign
// CHECK-NOT:     fir.call @_FortranACopyOutAssign(

// Assumed-rank keeps runtime allocation because its rank cannot be represented
// by inline fir.alloca/fir.allocmem operations. Deallocation remains inline.
func.func @test_assumed_rank_copy_pair(%box: !fir.box<!fir.array<*:f64>>, %temp: !fir.ref<!fir.box<!fir.array<*:f64>>>) {
  %0:3 = hlfir.copy_in %box to %temp : (!fir.box<!fir.array<*:f64>>, !fir.ref<!fir.box<!fir.array<*:f64>>>) -> (!fir.box<!fir.array<*:f64>>, i1, i1)
  hlfir.copy_out %temp, %0#1, %0#2 to %box : (!fir.ref<!fir.box<!fir.array<*:f64>>>, i1, i1, !fir.box<!fir.array<*:f64>>) -> ()
  return
}
// CHECK-LABEL: func.func @test_assumed_rank_copy_pair(
// CHECK:         fir.call @_FortranACopyInAssign
// CHECK:         fir.call @_FortranACopyOutAssignDirect
// CHECK:         fir.freemem {{.*}} : !fir.heap<i8>

func.func @test_copy_in_optional(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.array<?xf64>>>, %is_present: i1) {
  %0:3 = hlfir.copy_in %box to %temp handle_optional %is_present : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.array<?xf64>>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1, i1)
  return
}
// CHECK-LABEL: func.func @test_copy_in_optional(
// CHECK:         fir.if %{{.*}} -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:           fir.call @_FortranAShallowCopyDirect
// CHECK-NOT:     fir.call @_FortranACopyInAssign

func.func @test_copy_out_no_copy_back(%temp: !fir.ref<!fir.box<!fir.array<?xf64>>>, %was_copied: i1, %must_free: i1) {
  hlfir.copy_out %temp, %was_copied, %must_free : (!fir.ref<!fir.box<!fir.array<?xf64>>>, i1, i1) -> ()
  return
}
// CHECK-LABEL: func.func @test_copy_out_no_copy_back(
// CHECK-NOT:     fir.call @_FortranACopyOutAssignDirect
// CHECK:         fir.if %{{.*}} {
// CHECK:           fir.load %{{.*}} : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:           fir.freemem

func.func @test_copy_out_copy_back(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.array<?xf64>>>, %was_copied: i1, %must_free: i1) {
  hlfir.copy_out %temp, %was_copied, %must_free to %box : (!fir.ref<!fir.box<!fir.array<?xf64>>>, i1, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL: func.func @test_copy_out_copy_back(
// CHECK:         fir.if %{{.*}} {
// CHECK:           fir.call @_FortranACopyOutAssignDirect({{.*}}) : (!fir.box<none>, !fir.ref<!fir.box<none>>, !fir.ref<i8>, i32) -> ()
// CHECK:         fir.if %{{.*}} {
// CHECK:           fir.freemem

func.func @test_copy_in_poly(%poly : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, %temp: !fir.ref<!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>) {
  %0:3 = hlfir.copy_in %poly to %temp : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, !fir.ref<!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>) -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, i1, i1)
  return
}
// CHECK-LABEL: func.func @test_copy_in_poly(
// CHECK:         fir.call @_FortranACopyInAssign
