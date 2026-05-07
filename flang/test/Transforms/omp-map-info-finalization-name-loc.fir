// RUN: fir-opt --omp-map-info-finalization --mlir-print-debuginfo %s | FileCheck %s

#loc_decl = loc("omp-map-info-finalization-name-loc.fir":12:1)
#loc_name = loc("arr(1:42)"(#loc_decl))

func.func @preserve_name_loc_for_descriptor_base_member(%arg0: !fir.box<!fir.array<?xi32>>) {
  %0 = fir.alloca !fir.box<!fir.heap<i32>>
  %1 = fir.zero_bits !fir.heap<i32>
  %2:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "test"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.embox %1 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "test2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
  %5 = fir.allocmem i32 {fir.must_be_heap = true}
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %4#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 10 : index
  %dims:3 = fir.box_dims %2#1, %c1 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %bounds = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c2 : index) extent(%dims#1 : index) stride(%dims#2 : index) start_idx(%c0 : index) {stride_in_bytes = true}
  %7 = fir.box_addr %2#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %8 = omp.map.info var_ptr(%4#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>> loc(#loc_name)
  %9 = omp.map.info var_ptr(%7 : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>) map_clauses(from) capture(ByRef) bounds(%bounds) -> !fir.ref<!fir.array<?xi32>>
  omp.target map_entries(%8 -> %arg1, %9 -> %arg2 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>) {
    omp.terminator
  }
  return
}

// CHECK-LABEL: func.func @preserve_name_loc_for_descriptor_base_member
// CHECK: fir.box_offset {{.*}} base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>> loc(#loc[[LOCID:[0-9]+]])
// CHECK: omp.map.info {{.*}} map_clauses(tofrom){{.*}} loc(#loc[[LOCID]])
// CHECK: omp.map.info {{.*}} map_clauses(always, to){{.*}} members({{.*}}){{.*}} loc(#loc[[LOCID]])
// CHECK: #loc[[LOCID]] = loc("arr(1:42)"({{.*}}))
