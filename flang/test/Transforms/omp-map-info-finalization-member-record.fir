// RUN: fir-opt --omp-map-info-finalization %s | FileCheck %s

// Record type with an allocatable member.
!t1 = !fir.type<_QFTt1{a:!fir.box<!fir.heap<i32>>}>
// Record type containing t1.
!t2 = !fir.type<_QFTt2{b:!fir.type<_QFTt1{a:!fir.box<!fir.heap<i32>>}>}>

func.func @test_member_map_not_direct_target_operand(%arg0: !fir.ref<!t2>, %arg1: !fir.ref<!t2>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFEvar1"} : (!fir.ref<!t2>) -> (!fir.ref<!t2>, !fir.ref<!t2>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFEvar2"} : (!fir.ref<!t2>) -> (!fir.ref<!t2>, !fir.ref<!t2>)

  %b1 = hlfir.designate %0#0{"b"} : (!fir.ref<!t2>) -> !fir.ref<!t1>
  %b2 = hlfir.designate %1#0{"b"} : (!fir.ref<!t2>) -> !fir.ref<!t1>

  // Member maps for var1%b and var2%b. Type t1 is a record with allocatable
  // member, so isRecordWithAllocatableMember returns true for these. These
  // maps are NOT direct target operands — they are referenced only through the
  // parent map's members operand.
  // Without the fix, the pass would follow getFirstTargetUser through the parent
  // to the target, then call getOperandIndexForMap which asserts because the member
  // map is not in map_entries.
  %map_b1 = omp.map.info var_ptr(%b1 : !fir.ref<!t1>, !t1) map_clauses(to) capture(ByRef) -> !fir.ref<!t1> {name = "var1%b"}
  %map_b2 = omp.map.info var_ptr(%b2 : !fir.ref<!t1>, !t1) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!t1> {name = "var2%b"}

  // Parent maps for var1 and var2. (direct target operands)
  %map_var1 = omp.map.info var_ptr(%0#1 : !fir.ref<!t2>, !t2) map_clauses(to) capture(ByRef) members(%map_b1 : [0] : !fir.ref<!t1>) -> !fir.ref<!t2> {name = "var1", partial_map = true}
  %map_var2 = omp.map.info var_ptr(%1#1 : !fir.ref<!t2>, !t2) map_clauses(tofrom) capture(ByRef) members(%map_b2 : [0] : !fir.ref<!t1>) -> !fir.ref<!t2> {name = "var2", partial_map = true}

  // Target only lists parent maps as direct operands.
  omp.target map_entries(%map_var1 -> %a0, %map_var2 -> %a1 : !fir.ref<!t2>, !fir.ref<!t2>) {
    omp.terminator
  }
  return
}

// Member map are added to map_entries
// CHECK-LABEL: func.func @test_member_map_not_direct_target_operand
// CHECK: %[[MAP_B1:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}} {name = "var1%b"}
// CHECK: %[[MAP_B2:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {name = "var2%b"}
// CHECK: %[[MAP_VAR1:.*]] = omp.map.info {{.*}} members(%[[MAP_B1]] : [0] : {{.*}}) {{.*}} {name = "var1", partial_map = true}
// CHECK: %[[MAP_VAR2:.*]] = omp.map.info {{.*}} members(%[[MAP_B2]] : [0] : {{.*}}) {{.*}} {name = "var2", partial_map = true}
// CHECK: omp.target map_entries(%[[MAP_VAR1]] -> %{{.*}}, %[[MAP_VAR2]] -> %{{.*}}, %[[MAP_B1]] -> %{{.*}}, %[[MAP_B2]] -> %{{.*}} :
