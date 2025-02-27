// Tests that we implicitly map alloctable fields of a record when referenced in
// a target region.

// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

!record_t = !fir.type<_QFTrecord_t{
  not_to_implicitly_map:
    !fir.box<!fir.heap<!fir.array<?xf32>>>,
  to_implicitly_map:
    !fir.box<!fir.heap<!fir.array<?xf32>>>
}>

fir.global internal @_QFEdst_record : !record_t {
  %0 = fir.undefined !record_t
  fir.has_value %0 : !record_t
}

func.func @_QQmain() {
  %6 = fir.address_of(@_QFEdst_record) : !fir.ref<!record_t>
  %7:2 = hlfir.declare %6 {uniq_name = "_QFEdst_record"} : (!fir.ref<!record_t>) -> (!fir.ref<!record_t>, !fir.ref<!record_t>)
  %16 = omp.map.info var_ptr(%7#1 : !fir.ref<!record_t>, !record_t) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<!record_t> {name = "dst_record"}
  omp.target map_entries(%16 -> %arg0 : !fir.ref<!record_t>) {
    %20:2 = hlfir.declare %arg0 {uniq_name = "_QFEdst_record"} : (!fir.ref<!record_t>) -> (!fir.ref<!record_t>, !fir.ref<!record_t>)
    %23 = hlfir.designate %20#0{"to_implicitly_map"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!record_t>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.terminator
  }
  return
}

// CHECK: %[[RECORD_DECL:.*]]:2 = hlfir.declare %0 {uniq_name = "_QFEdst_record"}
// CHECK: %[[FIELD_COORD:.*]] = fir.coordinate_of %[[RECORD_DECL]]#1, %{{c1.*}}

// CHECK: %[[UPPER_BOUND:.*]] = arith.subi %{{.*}}#1, %{{c1.*}} : index

// CHECK: %[[BOUNDS:.*]] = omp.map.bounds 
// CHECK-SAME: lower_bound(%{{c0.*}} : index) upper_bound(%[[UPPER_BOUND]] : index)
// CHECK-SAME: extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index)
// CHECK-SAME: start_idx(%{{.*}}#0 : index) {stride_in_bytes = true}

// CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[FIELD_COORD]] base_addr
// CHECK: %[[FIELD_BASE_ADDR_MAP:.*]] = omp.map.info var_ptr(
// CHECK-SAME: %[[FIELD_COORD]] : {{.*}}) var_ptr_ptr(
// CHECK-SAME: %[[BASE_ADDR]] : {{.*}}) map_clauses(
// CHECK-SAME: implicit, tofrom) capture(ByRef) bounds(
// CHECK-SAME: %[[BOUNDS]])

// CHECK: %[[FIELD_MAP:.*]] = omp.map.info var_ptr(
// CHECK-SAME: %[[FIELD_COORD]] : {{.*}}) map_clauses(
// CHECK-SAME: implicit, to) capture(ByRef) ->
// CHECK-SAME: {{.*}} {name = "dst_record.to_implicitly_map.implicit_map"}

// CHECK: %[[RECORD_MAP:.*]] = omp.map.info var_ptr(
// CHECK-SAME: %[[RECORD_DECL]]#1 : {{.*}}) map_clauses(
// CHECK-SAME: implicit, tofrom) capture(ByRef) members(
// CHECK-SAME: %[[FIELD_MAP]], %[[FIELD_BASE_ADDR_MAP]] :
// CHECK-SAME: [1], [1, 0] : {{.*}}) -> {{.*}}> {name =
// CHECK-SAME: "dst_record", partial_map = true}

// CHECK: omp.target map_entries(
// CHECK-SAME: %[[RECORD_MAP]] -> %{{[^[:space:]]+}},
// CHECK-SAME: %[[FIELD_MAP]] -> %{{[^[:space:]]+}},
// CHECK-SAME: %[[FIELD_BASE_ADDR_MAP]] -> %{{[^[:space:]]+}}
// CHECK-SAME: : {{.*}})
