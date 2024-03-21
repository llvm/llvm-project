!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes="HLFIRDIALECT"

!HLFIRDIALECT: %[[POINTER:.*]]:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointer_routineEpoint"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!HLFIRDIALECT: %[[BOX_OFF:.*]] = fir.box_offset %[[POINTER]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
!HLFIRDIALECT: %[[POINTER_MAP_MEMBER:.*]] = omp.map.info var_ptr(%[[POINTER]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, i32) var_ptr_ptr(%[[BOX_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>)  map_clauses(implicit, tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!HLFIRDIALECT: %[[POINTER_MAP:.*]] = omp.map.info var_ptr(%[[POINTER]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(implicit, tofrom) capture(ByRef) members(%[[POINTER_MAP_MEMBER]] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "point"}
!HLFIRDIALECT: omp.target map_entries({{.*}}, %[[POINTER_MAP_MEMBER]] -> {{.*}}, %[[POINTER_MAP]] -> {{.*}} : {{.*}}, !fir.llvm_ptr<!fir.ref<i32>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
subroutine pointer_routine()
    integer, pointer :: point 
!$omp target map(tofrom:pointer)
    point = 1
!$omp end target
end subroutine pointer_routine
