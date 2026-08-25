! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Verify that we appropriately apply the close map type to the descriptor when it is
! explicitly specified and that we do not strip it needlessly in USM or otherwise add
! it to the descriptor implicitly like we used to.

module usm_descriptor_close_map
  !$omp requires unified_shared_memory
contains

! CHECK-LABEL: func.func @_QMusm_descriptor_close_mapPusm_only
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, {{.*}}Epoint"}
! CHECK: %[[BOX_OFF:.*]] = fir.box_offset %[[DECL]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK: %[[MEMBER:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) name("") -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(always, to) capture(ByRef) members(%[[MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("point") -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  subroutine usm_only(point)
    integer, pointer :: point
    !$omp target map(tofrom: point)
    point = 1
    !$omp end target
  end subroutine

! CHECK-LABEL: func.func @_QMusm_descriptor_close_mapPusm_explicit_close
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, {{.*}}Epoint"}
! CHECK: %[[BOX_OFF:.*]] = fir.box_offset %[[DECL]]#1 base_addr : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK: %[[MEMBER:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(close, tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>, i32) name("") -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.box<!fir.ptr<i32>>) map_clauses(close, to) capture(ByRef) members(%[[MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("point") -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  subroutine usm_explicit_close(point)
    integer, pointer :: point
    !$omp target map(close, tofrom: point)
    point = 1
    !$omp end target
  end subroutine

end module
