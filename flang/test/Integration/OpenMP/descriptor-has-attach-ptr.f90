! Verify that LLVM-IR lowering from Flang for descriptor-based maps drives
! OMPIRBuilder's HasAttachPtr logic correctly in generated mapper functions.

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa -mllvm -ignore-missing-type-desc -o - %s 2>&1 | FileCheck %s

module descriptor_has_attach_ptr_integ_m
  implicit none

  type :: dtype_a
    integer :: num_vals
    integer, allocatable :: var_a(:)
  end type dtype_a

  type :: mix_dtype
    integer :: scalar
    integer, allocatable :: arr(:)
  end type mix_dtype

  type :: pointee_t
    integer :: x
    integer :: y
  end type pointee_t

  type :: holder_t
    type(pointee_t), pointer :: p
  end type holder_t

contains

subroutine mapper_has_attach_ptr()
  type(dtype_a) :: s

  !$omp declare mapper (dtype_a :: var) map(var, var%var_a(1:var%num_vals))

  !$omp target map(mapper(default) : s)
  !$omp end target
end subroutine mapper_has_attach_ptr

subroutine mapper_implicit_mix(x)
  type(mix_dtype) :: x

  !$omp declare mapper (mix_dtype :: v) map(v, v%arr, v%scalar)

  !$omp target map(mapper(default) : x)
  !$omp end target

  !$omp target
  !$omp end target
end subroutine mapper_implicit_mix

subroutine mapper_member_of_has_attach_ptr()
  type(holder_t) :: h

  !$omp declare mapper(holder_t :: v) map(v, v%p%x, v%p%y)

  !$omp target map(mapper(default): h)
  !$omp end target
end subroutine mapper_member_of_has_attach_ptr

end module descriptor_has_attach_ptr_integ_m

! CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
! CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 288]

! CHECK-DAG: %.offload_mappers{{.*}} = alloca [2 x ptr]
! CHECK-DAG: store ptr @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_has_attach_ptrdtype_a_omp_default_mapper, ptr %{{.*}}, align 8
! CHECK-DAG: store ptr @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_implicit_mixmix_dtype_omp_default_mapper, ptr %{{.*}}, align 8
! CHECK-DAG: store ptr @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_member_of_has_attach_ptrholder_t_omp_default_mapper, ptr %{{.*}}, align 8

! CHECK-LABEL: define internal void @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_has_attach_ptrdtype_a_omp_default_mapper
! CHECK-SAME: (ptr noundef %{{.*}}, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}})
! CHECK: %[[DTA_PREV:.*]] = call i64 @__tgt_mapper_num_components
! CHECK: %[[DTA_SHIFT:.*]] = shl i64 %[[DTA_PREV]], 48

! Descriptor/object entry for var%var_a is an ordinary MEMBER_OF entry and is
! shifted by the current mapper component count.
! CHECK: %[[DTA_DESC_MEMBER:.*]] = add nuw i64 281474976710659, %[[DTA_SHIFT]]
! CHECK: %[[DTA_DESC_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc2 ], [ {{.*}}, %omp.type.to4 ], [ {{.*}}, %omp.type.from6 ], [ %[[DTA_DESC_MEMBER]], %omp.type.to.else5 ]
! CHECK: %[[DTA_DESC_WITH_MODS:.*]] = or i64 %[[DTA_DESC_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 56, i64 %[[DTA_DESC_WITH_MODS]], ptr @{{[0-9]+}})

! Pointee storage for var%var_a is HasAttachPtr without an inner MEMBER_OF, so
! it must not get an outer MEMBER_OF added.
! CHECK: %[[DTA_PTEE_TYPE:.*]] = phi i64 [ 0, %omp.type.alloc18 ], [ 1, %omp.type.to20 ], [ 2, %omp.type.from22 ], [ 3, %omp.type.to.else21 ]
! CHECK: %[[DTA_PTEE_WITH_MODS:.*]] = or i64 %[[DTA_PTEE_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[DTA_PTEE_WITH_MODS]], ptr @{{[0-9]+}})

! ATTACH remains raw ATTACH.
! CHECK: %[[DTA_ATTACH_TYPE:.*]] = phi i64 [ 16384, %omp.type.alloc26 ], [ 16384, %omp.type.to28 ], [ 16384, %omp.type.from30 ], [ 16384, %omp.type.to.else29 ]
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[DTA_ATTACH_TYPE]], ptr @{{[0-9]+}})

! CHECK-LABEL: define internal void @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_implicit_mixmix_dtype_omp_default_mapper
! CHECK-SAME: (ptr noundef %{{.*}}, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}})
! CHECK: %[[MIX_PREV:.*]] = call i64 @__tgt_mapper_num_components
! CHECK: %[[MIX_SHIFT:.*]] = shl i64 %[[MIX_PREV]], 48

! Scalar member is not a pointee reached through an attach pointer and is
! represented as a regular MEMBER_OF component.
! CHECK: %[[MIX_SCALAR_MEMBER:.*]] = add nuw i64 281474976710659, %[[MIX_SHIFT]]
! CHECK: %[[MIX_SCALAR_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc2 ], [ {{.*}}, %omp.type.to4 ], [ {{.*}}, %omp.type.from6 ], [ %[[MIX_SCALAR_MEMBER]], %omp.type.to.else5 ]
! CHECK: %[[MIX_SCALAR_WITH_MODS:.*]] = or i64 %[[MIX_SCALAR_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 {{.*}}, i64 %[[MIX_SCALAR_WITH_MODS]], ptr @{{[0-9]+}})

! CHECK-LABEL: define internal void @.omp_mapper._QQMdescriptor_has_attach_ptr_integ_mFmapper_member_of_has_attach_ptrholder_t_omp_default_mapper
! CHECK-SAME: (ptr noundef %{{.*}}, ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}})

! The mapper for holder_t contains p%x and p%y entries in the p pointee block.
! These entries are HasAttachPtr and already have an inner MEMBER_OF mask. The
! existing MEMBER_OF value is shifted by the number of pre-existing mapper components.
! CHECK: %[[H_PREV:.*]] = call i64 @__tgt_mapper_num_components
! CHECK: %[[H_SHIFT:.*]] = shl i64 %[[H_PREV]], 48

! Descriptor/object entry for p is an ordinary MEMBER_OF entry.
! CHECK: %[[H_DESC_MEMBER:.*]] = add nuw i64 281474976710659, %[[H_SHIFT]]
! CHECK: %[[H_DESC_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc1 ], [ {{.*}}, %omp.type.to3 ], [ {{.*}}, %omp.type.from5 ], [ %[[H_DESC_MEMBER]], %omp.type.to.else4 ]
! CHECK: %[[H_DESC_WITH_MODS:.*]] = or i64 %[[H_DESC_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 40, i64 %[[H_DESC_WITH_MODS]], ptr @{{[0-9]+}})

! p%x: HasAttachPtr and already MEMBER_OF the p pointee block.
! CHECK: %[[H_X_MEMBER:.*]] = add nuw i64 281474976710659, %[[H_SHIFT]]
! CHECK: %[[H_X_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc25 ], [ {{.*}}, %omp.type.to27 ], [ {{.*}}, %omp.type.from29 ], [ %[[H_X_MEMBER]], %omp.type.to.else28 ]
! CHECK: %[[H_X_WITH_MODS:.*]] = or i64 %[[H_X_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 4, i64 %[[H_X_WITH_MODS]], ptr @{{[0-9]+}})

! p%y: second HasAttachPtr entry that already has the same inner MEMBER_OF mask.
! CHECK: %[[H_Y_MEMBER:.*]] = add nuw i64 281474976710659, %[[H_SHIFT]]
! CHECK: %[[H_Y_TYPE:.*]] = phi i64 [ {{.*}}, %omp.type.alloc33 ], [ {{.*}}, %omp.type.to35 ], [ {{.*}}, %omp.type.from37 ], [ %[[H_Y_MEMBER]], %omp.type.to.else36 ]
! CHECK: %[[H_Y_WITH_MODS:.*]] = or i64 %[[H_Y_TYPE]], %{{.*}}
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 4, i64 %[[H_Y_WITH_MODS]], ptr @{{[0-9]+}})

! ATTACH for p remains raw ATTACH and is not HasAttachPtr.
! CHECK: %[[H_ATTACH_TYPE:.*]] = phi i64 [ 16384, %omp.type.alloc41 ], [ 16384, %omp.type.to43 ], [ 16384, %omp.type.from45 ], [ 16384, %omp.type.to.else44 ]
! CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %[[H_ATTACH_TYPE]], ptr @{{[0-9]+}})
