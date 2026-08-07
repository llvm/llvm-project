! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

program automap_program
   integer, allocatable, target :: automap_array(:)
   !$omp declare target enter(automap:automap_array)

   allocate (automap_array(10))

   !$omp target
      automap_array(1) = 1
   !$omp end target

   deallocate (automap_array)
end program

! CHECK-LABEL: func.func @_QQmain()
! CHECK-NOT: has_device_addr
! CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[DESC:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to, ref_ptr) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "_QFEautomap_array"}
! CHECK: %[[DESC_BASE:.*]] = fir.box_offset %[[DESC]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[ATTACH_MAP:.*]] = omp.map.info var_ptr(%[[DESC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr) capture(ByRef) var_ptr_ptr(%[[DESC_BASE]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "_QFEautomap_array"}
! CHECK: %[[STORAGE_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(storage) capture(ByRef) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "_QFEautomap_array"}
! CHECK: omp.target_enter_data map_entries(%[[DESC_MAP]], %[[STORAGE_MAP]], %[[ATTACH_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-NOT: has_device_addr
! CHECK: omp.target {{.*}}map_entries(
! CHECK-NOT: has_device_addr
! CHECK: %[[DELETE_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(delete) capture(ByRef) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "_QFEautomap_array"}
! CHECK: %[[DESC_DELETE:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(delete, attach_never, ref_ptr) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "_QFEautomap_array"}
! CHECK: omp.target_exit_data map_entries(%[[DELETE_MAP]], %[[DESC_DELETE]] : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
