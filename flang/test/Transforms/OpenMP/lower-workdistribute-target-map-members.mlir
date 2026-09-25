// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// splitTargetData must retarget a cloned map's "members" to the inner clone.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  func.func @map_members_split(%box : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %x : !fir.ref<i32>) {
    %base_off = fir.box_offset %box base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %data_map = omp.map.info var_ptr(%box : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%base_off : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.array<?xi32>) name("") -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %desc_map = omp.map.info var_ptr(%box : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%data_map : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %x_map = omp.map.info var_ptr(%x : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<i32>
    omp.target kernel_type(generic) map_entries(%data_map -> %arg_data, %desc_map -> %arg_desc, %x_map -> %arg_x : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<i32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %val = arith.constant 42 : i32
      omp.teams {
        omp.workdistribute {
          fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
            fir.store %val to %arg_x : !fir.ref<i32>
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL:   func.func @map_members_split(

// Original maps drive the outer target_data host<->device movement.
// CHECK:           %[[DATA_MAP:.*]] = omp.map.info {{.*}}map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:           %[[DESC_MAP:.*]] = omp.map.info {{.*}}map_clauses(to) capture(ByRef) members(%[[DATA_MAP]] : [0] : {{.*}}) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

// Inner clones are downgraded to "storage". The descriptor clone must point at
// the inner DATA clone, not the outer DATA_MAP above.
// CHECK:           %[[DATA_MAP_INNER:.*]] = omp.map.info {{.*}}map_clauses(storage) capture(ByRef) var_ptr_ptr({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:           %[[DESC_MAP_INNER:.*]] = omp.map.info {{.*}}map_clauses(storage) capture(ByRef) members(%[[DATA_MAP_INNER]] : [0] : {{.*}}) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

// CHECK:           omp.target_data map_entries({{.*}}%[[DATA_MAP]]{{.*}}%[[DESC_MAP]]{{.*}}) {
// CHECK:             omp.target {{.*}}map_entries({{.*}}%[[DATA_MAP_INNER]]{{.*}}%[[DESC_MAP_INNER]]{{.*}}) {
