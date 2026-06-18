// RUN: fir-opt --omp-automap-to-target-data %s | FileCheck %s
// Test OMP AutomapToTargetData pass.

module {
  fir.global
      @_QMtestEarr{omp.declare_target = #omp.declaretarget<device_type = (any),
                       capture_clause = (enter), automap = true>} target
                       : !fir.box<!fir.heap<!fir.array<?xi32>>>

  func.func @automap() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : i32
    %addr = fir.address_of(@_QMtestEarr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %decl:2 = hlfir.declare %addr {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMtestEarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
    %idx = fir.convert %c10 : (i32) -> index
    %cond = arith.cmpi sgt, %idx, %c0 : index
    %n = arith.select %cond, %idx, %c0 : index
    %mem = fir.allocmem !fir.array<?xi32>, %n {fir.must_be_heap = true}
    %shape = fir.shape %n : (index) -> !fir.shape<1>
    %box = fir.embox %mem(%shape) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %box to %decl#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %ld = fir.load %decl#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %base = fir.box_addr %ld : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    fir.freemem %base : !fir.heap<!fir.array<?xi32>>
    %undef = fir.zero_bits !fir.heap<!fir.array<?xi32>>
    %sh0 = fir.shape %c0 : (index) -> !fir.shape<1>
    %empty = fir.embox %undef(%sh0) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %empty to %decl#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    return
  }
}

// CHECK:         fir.global @[[AUTOMAP:.*]] {{{.*}} automap = true
// CHECK-LABEL:   func.func @automap()
// CHECK:           %[[AUTOMAP_ADDR:.*]] = fir.address_of(@[[AUTOMAP]])
// CHECK:           %[[AUTOMAP_DECL:.*]]:2 = hlfir.declare %[[AUTOMAP_ADDR]]
// CHECK:           %[[ALLOC_MEM:.*]] = fir.allocmem
// CHECK-NEXT:      fir.shape
// CHECK-NEXT:      %[[ARR_BOXED:.*]] = fir.embox %[[ALLOC_MEM]]
// CHECK-NEXT:      fir.store %[[ARR_BOXED]]
// CHECK:           %[[ARR_BOXED_LOADED:.*]] = fir.load %[[AUTOMAP_DECL]]#0
// CHECK:           %[[DIM0:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARR_BOXED_LOADED]], %[[DIM0]]
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_DIMS2:.*]]:3 = fir.box_dims %[[ARR_BOXED_LOADED]], %[[ZERO]]
// CHECK:           %[[LOWER_BOUND:.*]] = arith.constant 0 : index
// CHECK:           %[[UPPER_BOUND:.*]] = arith.subi %[[BOX_DIMS2]]#1, %[[ONE]] : index
// CHECK:           %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LOWER_BOUND]] : index) upper_bound(%[[UPPER_BOUND]] : index) extent(%[[BOX_DIMS2]]#1 : index) stride(%[[BOX_DIMS2]]#2 : index) start_idx(%[[BOX_DIMS]]#0 : index) {stride_in_bytes = true}
// CHECK:           %[[DATA_ADDR:.*]] = fir.box_addr %[[ARR_BOXED]]
// CHECK-NEXT:      %[[DATA_REF:.*]] = fir.convert %[[DATA_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK-NEXT:      %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[AUTOMAP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to, ref_ptr) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "_QMtestEarr"}
// CHECK-NEXT:      %[[STORAGE_MAP:.*]] = omp.map.info var_ptr(%[[DATA_REF]] : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(storage) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "_QMtestEarr"}
// CHECK-NEXT:      omp.target_enter_data map_entries(%[[DESC_MAP]], %[[STORAGE_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.array<?xi32>>)
// CHECK:           %[[LOAD:.*]] = fir.load %[[AUTOMAP_DECL]]#0
// CHECK:           %[[EXIT_ADDR:.*]] = fir.box_addr %[[LOAD]]
// CHECK-NEXT:      %[[EXIT_REF:.*]] = fir.convert %[[EXIT_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK-NEXT:      %[[EXIT_MAP:.*]] = omp.map.info var_ptr(%[[EXIT_REF]] : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(delete) capture(ByRef)
// CHECK-NEXT:      %[[DESC_DELETE:.*]] = omp.map.info var_ptr(%[[AUTOMAP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(delete, attach_never, ref_ptr) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "_QMtestEarr"}
// CHECK-NEXT:      omp.target_exit_data map_entries(%[[EXIT_MAP]], %[[DESC_DELETE]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK-NEXT:      %[[BOXADDR:.*]] = fir.box_addr %[[LOAD]]
// CHECK-NEXT:      fir.freemem %[[BOXADDR]]
