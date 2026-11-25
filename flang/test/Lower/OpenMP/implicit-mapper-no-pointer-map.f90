! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test that pointer components (including pointers to nested record types)
! do not have maps emitted inside implicit declare mappers, while
! allocatable components do have maps. Tests three levels of nesting.

program test_implicit_mapper_no_pointer_map
  implicit none

  type :: leaf_type
    integer              :: leaf_val = 0
    real(8), allocatable :: leaf_arr(:)
    real(8), pointer     :: leaf_ptr_arr(:) => null()  ! Should NOT be mapped
  end type leaf_type

  type :: inner_type
    integer              :: val = 0
    real(8), allocatable :: arr(:)
    real(8), pointer     :: ptr_arr(:) => null()          ! Should NOT be mapped
    type(leaf_type), allocatable :: alloc_leaf            ! SHOULD be mapped with nested mapper
    type(leaf_type), pointer     :: ptr_leaf => null()    ! Should NOT be mapped
  end type inner_type

  type :: outer_type
    integer                       :: id = 0
    type(inner_type), allocatable :: alloc_inner          ! SHOULD be mapped with nested mapper
    type(inner_type), pointer     :: ptr_inner => null()  ! Should NOT be mapped
    real(8), allocatable          :: alloc_arr(:)         ! SHOULD be mapped
    integer, pointer              :: ptr_scalar => null() ! Should NOT be mapped
  end type outer_type

  type(outer_type), allocatable :: obj

  !$omp target
    obj%id = 1
  !$omp end target

end program test_implicit_mapper_no_pointer_map

! The implicit default mapper transfers the record's flat storage (trivial
! members and descriptor bytes) via a single contiguous to/from parent map, and
! only emits ref_ptee (pointee + attach) entries for allocatable components.
! Trivial members (leaf_val, val, id) and pointer components get no member map.

! CHECK-LABEL: omp.declare_mapper @{{.*}}leaf_type_omp_default_mapper : !fir.type<_QFTleaf_type{
! CHECK: %[[LEAF_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom, ref_ptee){{.*}}-> !fir.llvm_ptr
! CHECK: %[[LEAF_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptee)
! CHECK: %[[LEAF_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTleaf_type{{.*}}>){{.*}}map_clauses(implicit, tofrom){{.*}}members(%[[LEAF_ARR_DATA]] : [1] :
! CHECK: omp.declare_mapper.info map_entries(%[[LEAF_PARENT]], %[[LEAF_ARR_DATA]], %[[LEAF_ARR_ATTACH]] :

! CHECK-LABEL: omp.declare_mapper @{{.*}}inner_type_omp_default_mapper : !fir.type<_QFTinner_type{
! CHECK: %[[INNER_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom, ref_ptee){{.*}}name(""){{.*}}-> !fir.llvm_ptr
! CHECK: %[[INNER_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptee){{.*}}name("")
! CHECK: %[[INNER_ALLOC_LEAF_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom, ref_ptee){{.*}}mapper(@{{.*}}leaf_type_omp_default_mapper){{.*}}name("")
! CHECK: %[[INNER_ALLOC_LEAF_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptee){{.*}}name("")
! CHECK: %[[INNER_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTinner_type{{.*}}>){{.*}}map_clauses(implicit, tofrom){{.*}}members(%[[INNER_ARR_DATA]], %[[INNER_ALLOC_LEAF_DATA]] : [1], [3] :
! CHECK: omp.declare_mapper.info map_entries(%[[INNER_PARENT]], %[[INNER_ARR_DATA]], %[[INNER_ALLOC_LEAF_DATA]], %[[INNER_ARR_ATTACH]], %[[INNER_ALLOC_LEAF_ATTACH]] :

! CHECK-LABEL: omp.declare_mapper @{{.*}}outer_type_omp_default_mapper : !fir.type<_QFTouter_type{
! CHECK: %[[OUTER_ALLOC_INNER_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom, ref_ptee){{.*}}mapper(@{{.*}}inner_type_omp_default_mapper){{.*}}name("")
! CHECK: %[[OUTER_ALLOC_INNER_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptee){{.*}}name("")
! CHECK: %[[OUTER_ALLOC_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom, ref_ptee){{.*}}name(""){{.*}}-> !fir.llvm_ptr
! CHECK: %[[OUTER_ALLOC_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptee){{.*}}name("")
! CHECK: %[[OUTER_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTouter_type{{.*}}>){{.*}}map_clauses(implicit, tofrom){{.*}}members(%[[OUTER_ALLOC_INNER_DATA]], %[[OUTER_ALLOC_ARR_DATA]] : [1], [3] :
! CHECK: omp.declare_mapper.info map_entries(%[[OUTER_PARENT]], %[[OUTER_ALLOC_INNER_DATA]], %[[OUTER_ALLOC_ARR_DATA]], %[[OUTER_ALLOC_INNER_ATTACH]], %[[OUTER_ALLOC_ARR_ATTACH]] :

! CHECK-LABEL: func.func @_QQmain
! CHECK: %[[DATA_MAP:.*]] = omp.map.info var_ptr({{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}mapper(@{{.*}}outer_type_omp_default_mapper){{.*}}-> !fir.llvm_ptr
! CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr({{.*}}){{.*}}map_clauses(always, implicit, to){{.*}}members(%[[DATA_MAP]] : [0] :{{.*}}){{.*}}name("obj")
! CHECK: %[[ATTACH_MAP:.*]] = omp.map.info var_ptr({{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}name("obj")
! CHECK: omp.target kernel_type(generic) map_entries(%[[DESC_MAP]] -> %{{.*}}, %[[ATTACH_MAP]] -> %{{.*}}, %[[DATA_MAP]] -> %{{.*}} :
