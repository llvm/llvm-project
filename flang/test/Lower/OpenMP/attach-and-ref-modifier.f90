! RUN: %flang_fc1 -fopenmp -fopenmp-version=61 -emit-hlfir %s -o - 2>&1 | FileCheck %s

subroutine attach_always()
    integer, pointer :: x

!CHECK: func.func @{{.*}}attach_always{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, to) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}}, %[[MAP_BASE_ADDR]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) {
    !$omp target map(attach(always): x)
        x = 1
    !$omp end target
end

subroutine attach_never()
    integer, pointer :: x

!CHECK: func.func @{{.*}}attach_never{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, to) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK-NOT: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_BASE_ADDR]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) {
    !$omp target map(attach(never): x)
        x = 1
    !$omp end target
end

subroutine attach_auto()
    integer, pointer :: x
!CHECK: func.func @{{.*}}attach_auto{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, to) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}}, %[[MAP_BASE_ADDR]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) {
    !$omp target map(attach(auto): x)
        x = 1
    !$omp end target
end

subroutine ref_ptr_ptee()
    integer, pointer :: x

!CHECK: func.func @{{.*}}ref_ptr_ptee{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptr_ptee) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(attach, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}}, %[[MAP_BASE_ADDR]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) {
    !$omp target map(ref_ptr_ptee, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptr()
  integer, pointer :: x

!CHECK: func.func @{{.*}}ref_ptr{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-NOT: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK-NOT: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses({{.*}}) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptr) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: %[[BASE_ADDR_2:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(attach, ref_ptr) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_2]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
    !$omp target map(ref_ptr, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptee()
  integer, pointer :: x

!CHECK: func.func @{{.*}}ref_ptee{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK-NOT: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses({{.*}}) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_BASE_ADDR]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
    !$omp target map(ref_ptee, to: x)
        x = 1
    !$omp end target
end

subroutine ref_ptr_ptee_attach_never()
    integer, pointer :: x

!CHECK: func.func @{{.*}}ref_ptr_ptee_attach_never{{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[BASE_ADDR_1:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK: %[[MAP_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptr_ptee) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_1]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[MAP_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(to, ref_ptr_ptee) capture(ByRef) members(%[[MAP_BASE_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK-NOT: %[[BASE_ADDR_2:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!CHECK-NOT: %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses({{.*}}) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_2]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "x"}
!CHECK: omp.target map_entries(%[[MAP_DESCRIPTOR]] -> %{{.*}}, %[[MAP_BASE_ADDR]] -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) {
    !$omp target map(attach(never), ref_ptr_ptee, to: x)
        x = 1
    !$omp end target
end
