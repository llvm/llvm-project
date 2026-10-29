! Verify the LLVM-IR effects of has_attach_ptr for descriptor/ref_ptr-only
! maps whose storage is inside another pointee. The inner descriptor map must
! not be made MEMBER_OF the outer mapped object when mapper lowering pushes
! components.

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa -mllvm -ignore-missing-type-desc -o - %s 2>&1 | FileCheck %s

module descriptor_has_attach_ptr_nested_pointee_integ_m
  implicit none

  type :: inner_t
    integer, pointer :: p(:)
  end type inner_t

  type :: outer_t
    type(inner_t), pointer :: mid
  end type outer_t

contains

subroutine nested_pointee_ref_ptr()
  type(outer_t) :: o

  !$omp declare mapper(outer_t :: v) map(v%mid, v%mid%p)

  !$omp target map(mapper(default): o)
  !$omp end target
end subroutine nested_pointee_ref_ptr

end module descriptor_has_attach_ptr_nested_pointee_integ_m

! CHECK-DAG: store ptr @.omp_mapper._QQMdescriptor_has_attach_ptr_nested_pointee_integ_mFnested_pointee_ref_ptrouter_t_omp_default_mapper, ptr %{{.*}}, align 8

! CHECK-LABEL: define internal void @.omp_mapper._QQMdescriptor_has_attach_ptr_nested_pointee_integ_mFnested_pointee_ref_ptrouter_t_omp_default_mapper
! CHECK-SAME: (ptr noundef %{{.*}}, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}})
! CHECK: %[[PREV:.*]] = call i64 @__tgt_mapper_num_components
! CHECK: %[[SHIFT:.*]] = shl i64 %[[PREV]], 48

! v%mid descriptor/object storage is in v's own storage, so mapper lowering
! adds MEMBER_OF for the mapper object.
! CHECK: %[[MID_DESC_MEMBER:.*]] = add nuw i64 281474976710661, %[[SHIFT]]
! CHECK: %[[MID_DESC_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc{{[0-9]+}} ], [ {{.*}}, %omp.type.to{{[0-9]+}} ], [ {{.*}}, %omp.type.from{{[0-9]+}} ], [ %[[MID_DESC_MEMBER]], %omp.type.to.else{{[0-9]+}} ]
! CHECK: %[[MID_DESC_WITH_MODS:.*]] = or i64 %[[MID_DESC_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 {{40|48}}, i64 %[[MID_DESC_WITH_MODS]], ptr @{{[0-9]+}})

! v%mid pointee storage is has_attach_ptr and must not receive a new outer
! MEMBER_OF bit. It is pushed with the raw to/from map type selected by the
! mapper component logic.
! CHECK: %[[MID_PTEE_TYPE:.*]] = phi i64 [ 0, %omp.type.alloc{{[0-9]+}} ], [ 1, %omp.type.to{{[0-9]+}} ], [ 2, %omp.type.from{{[0-9]+}} ], [ 3, %omp.type.to.else{{[0-9]+}} ]
! CHECK: %[[MID_PTEE_WITH_MODS:.*]] = or i64 %[[MID_PTEE_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[MID_PTEE_WITH_MODS]], ptr @{{[0-9]+}})

! v%mid%p descriptor/ref_ptr storage is physically inside the v%mid pointee.
! Because it is has_attach_ptr, mapper lowering must not add MEMBER_OF for the
! outer object.
! CHECK: %[[P_DESC_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc{{[0-9]+}} ], [ {{.*}}, %omp.type.to{{[0-9]+}} ], [ {{.*}}, %omp.type.from{{[0-9]+}} ], [ {{.*}}, %omp.type.to.else{{[0-9]+}} ]
! CHECK: %[[P_DESC_WITH_MODS:.*]] = or i64 %[[P_DESC_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 {{40|48}}, i64 %[[P_DESC_WITH_MODS]], ptr @{{[0-9]+}})

! v%mid%p pointee storage is also has_attach_ptr and remains independent of the
! outer object's ref-count.
! CHECK: %[[P_PTEE_TYPE:.*]] = phi i64 [ 0, %omp.type.alloc{{[0-9]+}} ], [ 1, %omp.type.to{{[0-9]+}} ], [ 2, %omp.type.from{{[0-9]+}} ], [ 3, %omp.type.to.else{{[0-9]+}} ]
! CHECK: %[[P_PTEE_WITH_MODS:.*]] = or i64 %[[P_PTEE_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[P_PTEE_WITH_MODS]], ptr @{{[0-9]+}})

! Both attach entries remain raw ATTACH and are not has_attach_ptr.
! CHECK: %[[MID_ATTACH_TYPE:.*]] = phi i64 [ 16384, %omp.type.alloc{{[0-9]+}} ], [ 16384, %omp.type.to{{[0-9]+}} ], [ 16384, %omp.type.from{{[0-9]+}} ], [ 16384, %omp.type.to.else{{[0-9]+}} ]
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[MID_ATTACH_TYPE]], ptr @{{[0-9]+}})
! CHECK: %[[P_ATTACH_TYPE:.*]] = phi i64 [ 16384, %omp.type.alloc{{[0-9]+}} ], [ 16384, %omp.type.to{{[0-9]+}} ], [ 16384, %omp.type.from{{[0-9]+}} ], [ 16384, %omp.type.to.else{{[0-9]+}} ]
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[P_ATTACH_TYPE]], ptr @{{[0-9]+}})
