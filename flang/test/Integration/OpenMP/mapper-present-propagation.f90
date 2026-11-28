!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -fopenmp -mllvm -ignore-missing-type-desc -fopenmp-version=60 -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=CHECK

! Verify PRESENT map-type modifier propagation through OpenMP mappers. Useful
! map-type modifier masks to verify correct path is generated:
!   ALWAYS | DELETE | CLOSE           = 0x40c  = 1036
!   ALWAYS | DELETE | CLOSE | PRESENT = 0x140c = 5132

program main
  implicit none

  type :: leaf_t
    integer :: i
    real :: r
    complex :: z
    integer :: fixed(4)
  end type

  type :: nested_pointee_t
    type(leaf_t) :: nested
    integer :: extra(2)
  end type

  type :: pointer_bundle_t
    integer :: n
    integer, pointer :: p(:)
    real, allocatable :: a(:)
    type(nested_pointee_t), pointer :: dp
    type(nested_pointee_t), allocatable :: da
  end type

  type :: mid_t
    type(leaf_t) :: leaf
    integer :: scalar
    real :: mat(2, 3)
    type(pointer_bundle_t) :: links
  end type

  type :: top_t
    integer :: tag
    real(8) :: weights(3)
    type(mid_t) :: left
    type(mid_t) :: right
  end type

!$omp declare mapper (top_mapper : top_t :: v) map (v%tag, v%weights, v%left%leaf%i, v%left%leaf%r, v%left%leaf%z, v%left%leaf%fixed, v%left%scalar, v%left%mat, v%left%links%n, v%left%links%p(1:v%left%links%n), v%left%links%a(1:v%left%links%n), v%left%links%dp, v%left%links%da, v%right%leaf%i, v%right%leaf%r, v%right%leaf%z, v%right%leaf%fixed, v%right%scalar, v%right%mat, v%right%links%n, v%right%links%p(1:v%right%links%n), v%right%links%a(1:v%right%links%n), v%right%links%dp, v%right%links%da, v%right%links%dp%extra, v%right%links%da%extra, v%right%links%dp%nested, v%right%links%da%nested, v%right%links%dp%nested%i, v%right%links%da%nested%i, v%right%links%dp%nested%r, v%right%links%da%nested%r, v%right%links%dp%nested%z, v%right%links%da%nested%z, v%right%links%dp%nested%fixed, v%right%links%da%nested%fixed)

  type(top_t) :: x

  allocate(x%right%links%da)
  allocate(x%right%links%dp)

!$omp target map(mapper(top_mapper), present, tofrom: x)
    x%tag = x%tag + 1
!$omp end target

end program

! CHECK-LABEL: define internal void @.omp_mapper.{{.*}}top_mapper

! Shared-storage entries in the mapper inherit PRESENT. This covers
! ordinary top-level members and nested derived-type members listed.

! CHECK: %[[TAG_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[TAG_WITH:.*]] = or i64 %{{.*}}, %[[TAG_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[TAG_WITH]],
! CHECK: %[[WEIGHTS_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[WEIGHTS_WITH:.*]] = or i64 %{{.*}}, %[[WEIGHTS_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[WEIGHTS_WITH]],
! CHECK: %[[WEIGHTS_ELEM_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[WEIGHTS_ELEM_WITH:.*]] = or i64 %{{.*}}, %[[WEIGHTS_ELEM_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[WEIGHTS_ELEM_WITH]],
! CHECK: %[[LEFT_LEAF_I_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_LEAF_I_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_LEAF_I_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_LEAF_I_WITH]],
! CHECK: %[[LEFT_LEAF_R_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_LEAF_R_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_LEAF_R_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_LEAF_R_WITH]],
! CHECK: %[[LEFT_LEAF_Z_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_LEAF_Z_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_LEAF_Z_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_LEAF_Z_WITH]],
! CHECK: %[[LEFT_FIXED_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_FIXED_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_FIXED_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_FIXED_WITH]],
! CHECK: %[[LEFT_SCALAR_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_SCALAR_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_SCALAR_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_SCALAR_WITH]],
! CHECK: %[[LEFT_MAT_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_MAT_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_MAT_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_MAT_WITH]],
! CHECK: %[[LEFT_LINKS_N_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LEFT_LINKS_N_WITH:.*]] = or i64 %{{.*}}, %[[LEFT_LINKS_N_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[LEFT_LINKS_N_WITH]],

! Pointee storage entries reached through pointer/allocatable components also inherit PRESENT.

! CHECK: %[[B60_POINTEE_TYPE:.*]] = phi i64 [ 0,
! CHECK: %[[B60_POINTEE_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[B60_POINTEE_WITH:.*]] = or i64 %[[B60_POINTEE_TYPE]], %[[B60_POINTEE_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[B60_POINTEE_WITH]],
! CHECK: %[[B60_POINTEE2_TYPE:.*]] = phi i64 [ 0,
! CHECK: %[[B60_POINTEE2_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[B60_POINTEE2_WITH:.*]] = or i64 %[[B60_POINTEE2_TYPE]], %[[B60_POINTEE2_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[B60_POINTEE2_WITH]],
! CHECK: %[[B60_DERIVED_PTR_TYPE:.*]] = phi i64 [ 0,
! CHECK: %[[B60_DERIVED_PTR_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[B60_DERIVED_PTR_WITH:.*]] = or i64 %[[B60_DERIVED_PTR_TYPE]], %[[B60_DERIVED_PTR_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[B60_DERIVED_PTR_WITH]],
! CHECK: %[[B60_DERIVED_ALLOC_TYPE:.*]] = phi i64 [ 0,
! CHECK: %[[B60_DERIVED_ALLOC_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[B60_DERIVED_ALLOC_WITH:.*]] = or i64 %[[B60_DERIVED_ALLOC_TYPE]], %[[B60_DERIVED_ALLOC_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[B60_DERIVED_ALLOC_WITH]],

! ATTACH entries do not have present applied.

! CHECK: %[[ATTACH_TYPE:.*]] = phi i64 [ 16384,
! CHECK: %[[ATTACH_MODS:.*]] = and i64 %{{.*}}, 1036
! CHECK: %[[ATTACH_WITH:.*]] = or i64 %[[ATTACH_TYPE]], %[[ATTACH_MODS]]
! CHECK: call void {{(@__tgt_push_mapper_component|@\.omp_mapper[^ ]*)}}({{.*}}i64 %[[ATTACH_TYPE]],

subroutine nested_allocatable_case()
  implicit none
 
  type :: bot_t
    integer :: bot_scalar
    integer, allocatable :: bot_scalar_a
  end type

  type :: mid_t
    integer :: mid_scalar
    integer, allocatable :: mid_scalar_a
    type(bot_t), allocatable :: link_bottom
  end type

  type :: top_t
    integer :: scalar
    type(mid_t), allocatable :: link_mid
  end type

!$omp declare mapper (nested_mapper : top_t :: v) map (v%scalar, v%link_mid, v%link_mid%mid_scalar_a, v%link_mid%mid_scalar, v%link_mid%link_bottom, v%link_mid%link_bottom%bot_scalar, v%link_mid%link_bottom%bot_scalar_a)

  type(top_t) :: x

  allocate(x%link_mid)
  allocate(x%link_mid%link_bottom)

!$omp target map(mapper(nested_mapper), present, tofrom: x)
!$omp end target

end subroutine nested_allocatable_case


! CHECK-LABEL: define internal void @.omp_mapper.{{.*}}nested_mapper

! The generated mapper should import PRESENT for every eligible non-ATTACH
! mapper component.  For this reduced nested-allocatable case, that includes
! shared storage in the top object, descriptor storage for allocatable members,
! and storage inside allocated nested derived-type objects.  Nested derived-type
! entries may lower either as direct mapper components or as nested default
! mapper calls; in either case, the effective type passed onward must include
! the imported PRESENT bit.


! CHECK: %[[TOP_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[TOP_WITH:.*]] = or i64 %{{.*}}, %[[TOP_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[TOP_WITH]],

! v%scalar
! CHECK: %[[SCALAR_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[SCALAR_WITH:.*]] = or i64 %{{.*}}, %[[SCALAR_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[SCALAR_WITH]],

! v%link_mid descriptor storage.
! CHECK: %[[LINK_MID_DESC_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LINK_MID_DESC_WITH:.*]] = or i64 %{{.*}}, %[[LINK_MID_DESC_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[LINK_MID_DESC_WITH]],

! v%link_mid lowered through the nested default mapper with PRESENT propagated.
! CHECK: %[[LINK_MID_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LINK_MID_WITH:.*]] = or i64 %{{.*}}, %[[LINK_MID_MODS]]
! CHECK: call void @.omp_mapper.{{.*}}mid_t_omp_default_mapper({{.*}}i64 %[[LINK_MID_WITH]],

! v%link_mid%mid_scalar_a descriptor storage.
! CHECK: %[[MID_SCALAR_A_DESC_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[MID_SCALAR_A_DESC_WITH:.*]] = or i64 %{{.*}}, %[[MID_SCALAR_A_DESC_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[MID_SCALAR_A_DESC_WITH]],

! v%link_mid%mid_scalar_a allocated storage.
! CHECK: %[[MID_SCALAR_A_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[MID_SCALAR_A_WITH:.*]] = or i64 %{{.*}}, %[[MID_SCALAR_A_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[MID_SCALAR_A_WITH]],

! v%link_mid%mid_scalar
! CHECK: %[[MID_SCALAR_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[MID_SCALAR_WITH:.*]] = or i64 %{{.*}}, %[[MID_SCALAR_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[MID_SCALAR_WITH]],

! v%link_mid%link_bottom descriptor storage.
! CHECK: %[[LINK_BOTTOM_DESC_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LINK_BOTTOM_DESC_WITH:.*]] = or i64 %{{.*}}, %[[LINK_BOTTOM_DESC_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[LINK_BOTTOM_DESC_WITH]],

! v%link_mid%link_bottom lowered through the nested default mapper.
! CHECK: %[[LINK_BOTTOM_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[LINK_BOTTOM_WITH:.*]] = or i64 %{{.*}}, %[[LINK_BOTTOM_MODS]]
! CHECK: call void @.omp_mapper.{{.*}}bot_t_omp_default_mapper({{.*}}i64 %[[LINK_BOTTOM_WITH]],

! v%link_mid%link_bottom%bot_scalar
! CHECK: %[[BOT_SCALAR_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[BOT_SCALAR_WITH:.*]] = or i64 %{{.*}}, %[[BOT_SCALAR_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[BOT_SCALAR_WITH]],

! v%link_mid%link_bottom%bot_scalar_a descriptor storage.
! CHECK: %[[BOT_SCALAR_A_DESC_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[BOT_SCALAR_A_DESC_WITH:.*]] = or i64 %{{.*}}, %[[BOT_SCALAR_A_DESC_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[BOT_SCALAR_A_DESC_WITH]],

! v%link_mid%link_bottom%bot_scalar_a allocated storage.
! CHECK: %[[BOT_SCALAR_A_MODS:.*]] = and i64 %{{.*}}, 5132
! CHECK: %[[BOT_SCALAR_A_WITH:.*]] = or i64 %{{.*}}, %[[BOT_SCALAR_A_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[BOT_SCALAR_A_WITH]],

! ATTACH entries are excluded from present map type propagation.
! CHECK: %[[ATTACH_MID_TYPE:.*]] = phi i64 [ 16384,
! CHECK: %[[ATTACH_MID_MODS:.*]] = and i64 %{{.*}}, 1036
! CHECK: %[[ATTACH_MID_WITH:.*]] = or i64 %[[ATTACH_MID_TYPE]], %[[ATTACH_MID_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[ATTACH_MID_TYPE]],

! CHECK: %[[ATTACH_MID_A_TYPE:.*]] = phi i64 [ 16384,
! CHECK: %[[ATTACH_MID_A_MODS:.*]] = and i64 %{{.*}}, 1036
! CHECK: %[[ATTACH_MID_A_WITH:.*]] = or i64 %[[ATTACH_MID_A_TYPE]], %[[ATTACH_MID_A_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[ATTACH_MID_A_TYPE]],

! CHECK: %[[ATTACH_BOTTOM_TYPE:.*]] = phi i64 [ 16384,
! CHECK: %[[ATTACH_BOTTOM_MODS:.*]] = and i64 %{{.*}}, 1036
! CHECK: %[[ATTACH_BOTTOM_WITH:.*]] = or i64 %[[ATTACH_BOTTOM_TYPE]], %[[ATTACH_BOTTOM_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[ATTACH_BOTTOM_TYPE]],

! CHECK: %[[ATTACH_BOT_A_TYPE:.*]] = phi i64 [ 16384,
! CHECK: %[[ATTACH_BOT_A_MODS:.*]] = and i64 %{{.*}}, 1036
! CHECK: %[[ATTACH_BOT_A_WITH:.*]] = or i64 %[[ATTACH_BOT_A_TYPE]], %[[ATTACH_BOT_A_MODS]]
! CHECK: call void @__tgt_push_mapper_component({{.*}}i64 %[[ATTACH_BOT_A_TYPE]],
