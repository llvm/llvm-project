! Verify that descriptor/ref_ptr-only maps whose storage is inside another
! pointee are marked has_attach_ptr. This is distinct from ordinary descriptor
! object maps in the mapped object's own storage.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module descriptor_has_attach_ptr_nested_pointee_m
  implicit none

  type :: inner_t
    integer, pointer :: p(:)
  end type inner_t

  type :: outer_t
    type(inner_t), pointer :: mid
  end type outer_t

  type :: pointee_t
    integer :: x
  end type pointee_t

contains

subroutine nested_pointee_ref_ptr(o)
  type(outer_t) :: o

!$omp declare mapper(outer_t :: v) map(v%mid, v%mid%p)

! CHECK-LABEL: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_nested_pointee_mFnested_pointee_ref_ptrouter_t_omp_default_mapper
! CHECK: %[[MID_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>) name("v%mid") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>
! CHECK: %[[MID_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>) map_clauses(always, to) capture(ByRef) name("v%mid") -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>>
! CHECK: %[[MID_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>) name("v%mid") -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTinner_t{{.*}}>>>>
! CHECK: %[[P_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("v%mid%p") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[P_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) name("v%mid%p") has_attach_ptr(true) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[P_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("v%mid%p") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTouter_t{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTouter_t{{.*}}>) map_clauses(storage) capture(ByRef) members(%[[MID_DESC]], %[[MID_PTEE]], %[[P_DESC]], %[[P_PTEE]] : [0], [0, 0], [0, 0, 0], [0, 0, 0, 0] : {{.*}}) name("v") partial_map(true) -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTouter_t{{.*}}>>
! CHECK: omp.declare_mapper.info map_entries(%[[PARENT]], %[[MID_DESC]], %[[P_DESC]], %[[MID_ATTACH]], %[[P_ATTACH]], %[[MID_PTEE]], %[[P_PTEE]] : {{.*}})

!$omp target map(mapper(default): o)
!$omp end target

end subroutine nested_pointee_ref_ptr

subroutine top_level_ref_ptee_member(p)
  type(pointee_t), pointer :: p

!$omp target map(ref_ptee, to: p) map(to: p%x)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_nested_pointee_mPtop_level_ref_ptee_member
! CHECK: %[[PX:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) name("p%x") has_attach_ptr(true) -> !fir.ref<i32>
! CHECK: %[[P_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>) map_clauses(to, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>) members(%[[PX]] : [0] : {{.*}}) name("p") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>
! CHECK: %[[P_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>) name("p") -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_nested_pointee_mTpointee_t{{.*}}>>>>
! CHECK: omp.target kernel_type(generic) map_entries(%[[P_PTEE]] -> {{.*}}, %[[PX]] -> {{.*}}, %[[P_ATTACH]] -> {{.*}} : {{.*}})

end subroutine top_level_ref_ptee_member

end module descriptor_has_attach_ptr_nested_pointee_m
