// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

// Test that the 'close' map flag is cleared from member maps if the parent map
// (derived type) does not have the 'close' flag. This typically happens in
// Unified Shared Memory (USM) mode where the parent is in USM (no close) but
// members (like descriptors) might have been initially tagged with close.

module attributes {omp.requires = #omp<clause_requires unified_shared_memory>} {
  func.func @test_usm_close_flag_cleanup(%arg0: !fir.ref<!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
    %map = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> {name = "parent"}

    omp.target kernel_type(generic) map_entries(%map -> %arg1 : !fir.ref<!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
      // Simulate usage to trigger implicit map addition
      %1 = hlfir.designate %arg1{"a"} : (!fir.ref<!fir.type<t{a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_usm_close_flag_cleanup
// CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} {name = "parent.a.implicit_map"}
// CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} members(%[[MEMBER]], {{.*}}) {{.*}} {name = "parent", {{.*}}}
// CHECK-NOT: close
