! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Test that pointer components (including pointers to nested record types)
! do not have maps emitted inside explicit default mapper calls, while
! allocatable components do have maps. The pointer can still be mapped
! explicitly via a separate map clause.

program test_default_mapper_no_pointer_map
  implicit none

  type :: leaf_type
    integer              :: leaf_val = 0
    real(8), allocatable :: leaf_arr(:)
    real(8), pointer     :: leaf_ptr_arr(:) => null()  ! Should NOT be mapped by mapper
  end type leaf_type

  type :: inner_type
    integer              :: val = 0
    real(8), allocatable :: arr(:)
    real(8), pointer     :: ptr_arr(:) => null()          ! Should NOT be mapped by mapper
    type(leaf_type), allocatable :: alloc_leaf            ! SHOULD be mapped with nested mapper
    type(leaf_type), pointer     :: ptr_leaf => null()    ! Should NOT be mapped by mapper
  end type inner_type

  type :: outer_type
    integer                       :: id = 0
    type(inner_type), allocatable :: alloc_inner          ! SHOULD be mapped with nested mapper
    type(inner_type), pointer     :: ptr_inner => null()  ! Should NOT be mapped by mapper
    real(8), allocatable          :: alloc_arr(:)         ! SHOULD be mapped
    integer, pointer              :: ptr_scalar => null() ! Should NOT be mapped by mapper, but CAN be mapped explicitly
  end type outer_type

  type(outer_type) :: obj

  !$omp target map(mapper(default), tofrom: obj) map(tofrom: obj%ptr_scalar)
    obj%id = obj%id + 1
  !$omp end target

end program test_default_mapper_no_pointer_map

! CHECK-LABEL: omp.declare_mapper @{{.*}}leaf_type_omp_default_mapper : !fir.type<_QFTleaf_type{
! CHECK: %[[LEAF_VAL:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32){{.*}}map_clauses(implicit, tofrom)
! CHECK: %[[LEAF_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}-> !fir.llvm_ptr
! CHECK: %[[LEAF_ARR_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, implicit, to)
! CHECK: %[[LEAF_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee)
! CHECK: %[[LEAF_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTleaf_type{{.*}}>){{.*}}members(%[[LEAF_VAL]], %[[LEAF_ARR_DESC]], %[[LEAF_ARR_DATA]] : [0], [1], [1, 0] :
! CHECK: omp.declare_mapper.info map_entries(%[[LEAF_PARENT]], %[[LEAF_VAL]], %[[LEAF_ARR_DESC]], %[[LEAF_ARR_ATTACH]], %[[LEAF_ARR_DATA]] :

! CHECK-LABEL: omp.declare_mapper @{{.*}}inner_type_omp_default_mapper : !fir.type<_QFTinner_type{
! CHECK: %[[INNER_VAL:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32){{.*}}map_clauses(implicit, tofrom)
! CHECK: %[[INNER_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}-> !fir.llvm_ptr{{.*}}{name = ""}
! CHECK: %[[INNER_ARR_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, implicit, to){{.*}}{name = ""}
! CHECK: %[[INNER_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}{name = ""}
! CHECK: %[[INNER_ALLOC_LEAF_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}mapper(@{{.*}}leaf_type_omp_default_mapper){{.*}}-> !fir.llvm_ptr
! CHECK: %[[INNER_ALLOC_LEAF_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, implicit, to){{.*}}{name = ""}
! CHECK: %[[INNER_ALLOC_LEAF_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}{name = ""}
! CHECK: %[[INNER_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTinner_type{{.*}}>){{.*}}members(%[[INNER_VAL]], %[[INNER_ARR_DESC]], %[[INNER_ARR_DATA]], %[[INNER_ALLOC_LEAF_DESC]], %[[INNER_ALLOC_LEAF_DATA]] : [0], [1], [1, 0], [3], [3, 0] :
! CHECK: omp.declare_mapper.info map_entries(%[[INNER_PARENT]], %[[INNER_VAL]], %[[INNER_ARR_DESC]], %[[INNER_ALLOC_LEAF_DESC]], %[[INNER_ARR_ATTACH]], %[[INNER_ALLOC_LEAF_ATTACH]], %[[INNER_ARR_DATA]], %[[INNER_ALLOC_LEAF_DATA]] :

! CHECK-LABEL: omp.declare_mapper @{{.*}}outer_type_omp_default_mapper : !fir.type<_QFTouter_type{
! CHECK: %[[OUTER_ID:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32){{.*}}map_clauses(implicit, tofrom)
! CHECK: %[[OUTER_ALLOC_INNER_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}mapper(@{{.*}}inner_type_omp_default_mapper){{.*}}-> !fir.llvm_ptr
! CHECK: %[[OUTER_ALLOC_INNER_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, implicit, to){{.*}}{name = ""}
! CHECK: %[[OUTER_ALLOC_INNER_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}{name = ""}
! CHECK: %[[OUTER_ALLOC_ARR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(implicit, tofrom){{.*}}-> !fir.llvm_ptr{{.*}}{name = ""}
! CHECK: %[[OUTER_ALLOC_ARR_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, implicit, to){{.*}}{name = ""}
! CHECK: %[[OUTER_ALLOC_ARR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}{name = ""}
! CHECK: %[[OUTER_PARENT:.*]] = omp.map.info var_ptr({{.*}}!fir.type<_QFTouter_type{{.*}}>){{.*}}members(%[[OUTER_ID]], %[[OUTER_ALLOC_INNER_DESC]], %[[OUTER_ALLOC_INNER_DATA]], %[[OUTER_ALLOC_ARR_DESC]], %[[OUTER_ALLOC_ARR_DATA]] : [0], [1], [1, 0], [3], [3, 0] :
! CHECK: omp.declare_mapper.info map_entries(%[[OUTER_PARENT]], %[[OUTER_ID]], %[[OUTER_ALLOC_INNER_DESC]], %[[OUTER_ALLOC_ARR_DESC]], %[[OUTER_ALLOC_INNER_ATTACH]], %[[OUTER_ALLOC_ARR_ATTACH]], %[[OUTER_ALLOC_INNER_DATA]], %[[OUTER_ALLOC_ARR_DATA]] :

! CHECK-LABEL: func.func @_QQmain
! CHECK: %[[PTR_SCALAR_DATA:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(tofrom){{.*}}-> !fir.llvm_ptr
! CHECK: %[[PTR_SCALAR_DESC:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(always, to){{.*}}{name = "obj%ptr_scalar"}
! CHECK: %[[PTR_SCALAR_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}{name = "obj%ptr_scalar"}
! CHECK: %[[OBJ_MAP:.*]] = omp.map.info var_ptr(%{{.*}}){{.*}}map_clauses(tofrom){{.*}}members(%[[PTR_SCALAR_DESC]], %[[PTR_SCALAR_DATA]] : [4], [4, 0] :{{.*}}){{.*}}{name = "obj"}
! CHECK: omp.target kernel_type(generic) map_entries(%[[OBJ_MAP]] -> %{{.*}}, %[[PTR_SCALAR_DESC]] -> %{{.*}}, %[[PTR_SCALAR_ATTACH]] -> %{{.*}}, %[[PTR_SCALAR_DATA]] -> %{{.*}} :
