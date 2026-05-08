// RUN: fir-opt --omp-map-info-finalization %s | FileCheck %s
// This tests that we are appropriately handling derived type mappings in the
// MapInfoFinalization pass, making sure that we connect the members contained
// by the parent to the target region using the parent, before processing
// further.
//
// The initial lowering just inserts the members into the parent, it's up to the
// MapInfoFinalization pass to connect these members (and any expanded members)
// to the targets block arguments, this is primarily done for canonicalization of
// the target region input arguments. However, some steps in the MapInfoFinalization
// pass depend on this canon representation (primarily the implicit mapping), so we
// must make sure this step occurs a the beginning and end of the pass to prevent
// issues.

!t1 = !fir.type<_QFTt1{a:!fir.box<!fir.heap<i32>>}>
!t2 = !fir.type<_QFTt2{b:!fir.type<_QFTt1{a:!fir.box<!fir.heap<i32>>}>}>

func.func @test_member_map_not_direct_target_operand(%arg0: !fir.ref<!t2>, %arg1: !fir.ref<!t2>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFEvar1"} : (!fir.ref<!t2>) -> (!fir.ref<!t2>, !fir.ref<!t2>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFEvar2"} : (!fir.ref<!t2>) -> (!fir.ref<!t2>, !fir.ref<!t2>)

  %b1 = hlfir.designate %0#0{"b"} : (!fir.ref<!t2>) -> !fir.ref<!t1>
  %b2 = hlfir.designate %1#0{"b"} : (!fir.ref<!t2>) -> !fir.ref<!t1>

  %map_b1 = omp.map.info var_ptr(%b1 : !fir.ref<!t1>, !t1) map_clauses(to) capture(ByRef) -> !fir.ref<!t1> {name = "var1%b"}
  %map_b2 = omp.map.info var_ptr(%b2 : !fir.ref<!t1>, !t1) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!t1> {name = "var2%b"}

  %map_var1 = omp.map.info var_ptr(%0#1 : !fir.ref<!t2>, !t2) map_clauses(to) capture(ByRef) members(%map_b1 : [0] : !fir.ref<!t1>) -> !fir.ref<!t2> {name = "var1", partial_map = true}
  %map_var2 = omp.map.info var_ptr(%1#1 : !fir.ref<!t2>, !t2) map_clauses(tofrom) capture(ByRef) members(%map_b2 : [0] : !fir.ref<!t1>) -> !fir.ref<!t2> {name = "var2", partial_map = true}

  omp.target map_entries(%map_var1 -> %a0, %map_var2 -> %a1 : !fir.ref<!t2>, !fir.ref<!t2>) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @test_member_map_not_direct_target_operand
// CHECK: %[[MAP_B1:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} {name = "var1%b"}
// CHECK: %[[MAP_B2:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {name = "var2%b"}
// CHECK: %[[MAP_VAR1:.*]] = omp.map.info {{.*}} members(%[[MAP_B1]] : [0] : {{.*}}) {{.*}} {name = "var1", partial_map = true}
// CHECK: %[[MAP_VAR2:.*]] = omp.map.info {{.*}} members(%[[MAP_B2]] : [0] : {{.*}}) {{.*}} {name = "var2", partial_map = true}
// CHECK: omp.target map_entries(%[[MAP_VAR1]] -> %{{.*}}, %[[MAP_VAR2]] -> %{{.*}}, %[[MAP_B1]] -> %{{.*}}, %[[MAP_B2]] -> %{{.*}} :