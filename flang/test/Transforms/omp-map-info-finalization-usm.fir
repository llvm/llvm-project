// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

// Test that the 'close' map flag is cleared from member maps if the parent map
// does not have the 'close' flag. This typically happens in USM mode.
// In this test, the parent is a fir.array of derived type.
module attributes {omp.requires = #omp<clause_requires unified_shared_memory>} {
  func.func @test_usm_close_flag_cleanup_array(%arg0: !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) {
    // The implicit descriptor map for the member, with 'close'
    %member = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) map_clauses(always, close, descriptor, to) capture(ByRef) -> !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>> {name = "parent.a.implicit_map"}

    // The parent map, which is a fir.array, without 'close'
    %map = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) map_clauses(to) capture(ByRef) members(%member : [0] : !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) -> !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>> {name = "parent"}

    omp.target map_entries(%map -> %arg1 : !fir.ref<!fir.array<10x!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_usm_close_flag_cleanup_array
// CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} {name = "parent{{.*}}implicit_map"}
// CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} members(%[[MEMBER]]{{.*}}) {{.*}} {name = "parent"}
// CHECK-NOT: close
