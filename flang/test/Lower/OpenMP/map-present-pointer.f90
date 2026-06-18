! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine map_present_pointer(p)
  integer, pointer :: p(:)

! CHECK-LABEL: func.func @_QPmap_present_pointer
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>
! CHECK: %[[BASE:.*]] = fir.box_offset %[[DECL]]#1 base_addr
! CHECK: %[[PTEE_MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : {{.*}}) map_clauses(present, to) capture(ByRef) var_ptr_ptr(%[[BASE]] : {{.*}}) {{.*}} -> !fir.llvm_ptr
! CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : {{.*}}) map_clauses(always, to) capture(ByRef) members(%[[PTEE_MAP]] : [0] : {{.*}}) -> {{.*}} {name = "p"}
! CHECK: %[[ATTACH_MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : {{.*}}) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BASE]] : {{.*}})
! CHECK: omp.target map_entries(%[[DESC_MAP]] -> {{.*}}, %[[ATTACH_MAP]] -> {{.*}}, %[[PTEE_MAP]] -> {{.*}}
!$omp target map(present, to: p)
  if (associated(p)) p(1) = p(1)
!$omp end target
end subroutine
