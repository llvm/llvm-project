! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s --check-prefixes=CHECK,ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %s -o - | FileCheck %s --check-prefixes=CHECK,OFF

! Check that generated default mapper carriers are emitted for multi-level
! derived type hierarchies when nested components have user-defined default
! mappers. This must work both with and without
! -fno-implicit-default-mapper-allocatable-members.
!
! The compiler-generated top_t mapper should attach the user-defined middle_t
! default mapper to top%mid, and the user-defined middle_t default mapper should
! attach the user-defined leaf_t default mapper to middle%leaf.
!
! Also check a top-level allocatable component in a generated carrier mapper:
! default/on emits the allocatable member map, while the -fno flag suppresses
! only that allocatable member and still emits the nested default mapper carrier.

program main
  implicit none

  type :: leaf_t(k)
    integer, kind :: k
    integer(k), pointer :: p(:)
  end type

  !$omp declare mapper(default: leaf_t(4) :: v) map(tofrom: v%p)

  type :: middle_t
    integer :: payload(8)
    type(leaf_t(4)) :: leaf
  end type

  !$omp declare mapper(default: middle_t :: v) map(tofrom: v%leaf)

  type :: top_t
    integer :: scalar
    type(middle_t) :: mid
  end type

  type(top_t) :: x

  !$omp target map(tofrom: x)
    x%mid%leaf%p(1) = 42
  !$omp end target
end program

subroutine top_allocatable_component
  implicit none

  type :: leaf_alloc_t(k)
    integer, kind :: k
    integer(k), pointer :: p(:)
  end type

  !$omp declare mapper(default: leaf_alloc_t(4) :: v) map(tofrom: v%p)

  type :: middle_alloc_t
    integer :: payload(8)
    type(leaf_alloc_t(4)) :: leaf
  end type

  !$omp declare mapper(default: middle_alloc_t :: v) map(tofrom: v%leaf)

  type :: top_alloc_t
    integer, allocatable :: alloc_field(:)
    type(middle_alloc_t) :: mid
  end type

  type(top_alloc_t) :: x

  !$omp target map(tofrom: x)
    x%mid%leaf%p(1) = 42
  !$omp end target
end subroutine

! The allocatable variant's generated middle mapper carries the default leaf
! mapper, and the generated top mapper carries the default middle mapper.
! CHECK: omp.declare_mapper @{{.*middle_alloc_t_omp_default_mapper}}
! CHECK: omp.map.info {{.*}}mapper(@{{.*leaf_alloc_t.*_omp_default_mapper}})
! CHECK: omp.declare_mapper @{{.*top_alloc_t_omp_default_mapper}}

! Default/on maps the top-level allocatable member inside the generated mapper.
! ON: fir.field_index alloc_field
! ON: omp.map.info {{.*}}map_clauses(implicit, tofrom, ref_ptee)
! ON: fir.field_index mid
! ON: omp.map.info {{.*}}mapper(@{{.*middle_alloc_t_omp_default_mapper}})
! ON: omp.declare_mapper.info

! With -fno-implicit-default-mapper-allocatable-members, the generated carrier
! mapper omits the allocatable member map but still attaches the nested default
! mapper to mid.
! OFF-NOT: fir.field_index alloc_field
! OFF: fir.field_index mid
! OFF: omp.map.info {{.*}}mapper(@{{.*middle_alloc_t_omp_default_mapper}})
! OFF: omp.declare_mapper.info

! The user-defined middle_alloc_t mapper also carries the leaf_alloc_t mapper.
! CHECK: omp.declare_mapper @{{.*middle_alloc_t_omp_default_mapper}}
! CHECK: omp.map.info {{.*}}mapper(@{{.*leaf_alloc_t.*_omp_default_mapper}})

! The non-allocatable variant's generated middle mapper carries the default leaf
! mapper, and the generated top mapper carries the default middle mapper.
! CHECK: omp.declare_mapper @{{.*middle_t_omp_default_mapper}}
! CHECK: omp.map.info {{.*}}mapper(@{{.*leaf_t.*_omp_default_mapper}})
! CHECK: omp.declare_mapper @{{.*top_t_omp_default_mapper}}
! CHECK: omp.map.info {{.*}}mapper(@{{.*middle_t_omp_default_mapper}})

! The target maps use the generated top-level carrier mappers.
! CHECK-DAG: omp.map.info {{.*}}mapper(@{{.*top_t_omp_default_mapper}}) name("x")
! CHECK-DAG: omp.map.info {{.*}}mapper(@{{.*top_alloc_t_omp_default_mapper}}) name("x")
