! This test checks that descriptor-based maps get the has_attach_ptr marker
! set correctly on pointee-storage entries (base address + members) and not
! on ATTACH entries or descriptor maps themselves.
!
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

module descriptor_has_attach_ptr_m
  implicit none

  type :: dtype_a
    integer, allocatable :: var_a(:)
  end type dtype_a

  type :: inner_t
    integer, allocatable :: arr(:)
  end type inner_t

  type :: outer_t
    type(inner_t), allocatable :: comp(:)
  end type outer_t

  type :: scalar_block_t
    integer :: scalar
    integer, allocatable :: array(:)
  end type scalar_block_t

  type :: container_t
    type(scalar_block_t), allocatable :: blk
  end type container_t

  type :: level3_t
    integer, allocatable :: z(:)
  end type level3_t

  type :: level2_t
    type(level3_t), allocatable :: l3(:)
  end type level2_t

  type :: level1_t
    type(level2_t), allocatable :: l2(:)
  end type level1_t

  type :: map_dtype_t
    integer :: scalar
    integer, allocatable :: arr(:)
  end type map_dtype_t

  type :: exp_map_t
    type(scalar_block_t), allocatable :: blk
    real :: scalar_f
  end type exp_map_t

! CHECK: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_mFmapper_and_implicitmap_dtype_t_omp_default_mapper
! CHECK: %[[MIX_ARR_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("v%arr") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[MIX_ARR_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) name("v%arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[MIX_ARR_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("v%arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[MIX_SCALAR:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("v%scalar") -> !fir.ref<i32>
! CHECK: %[[MIX_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTmap_dtype_t{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_mTmap_dtype_t{{.*}}>) map_clauses(tofrom) capture(ByRef) members(%[[MIX_ARR_DESC]], %[[MIX_ARR_PTEE]], %[[MIX_SCALAR]] : [1], [1, 0], [0] : {{.*}}) name("v") -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTmap_dtype_t{{.*}}>>
! CHECK: omp.declare_mapper.info map_entries(%[[MIX_PARENT]], %[[MIX_ARR_DESC]], %[[MIX_SCALAR]], %[[MIX_ARR_ATTACH]], %[[MIX_ARR_PTEE]] : {{.*}})

! CHECK: omp.declare_mapper @_QMdescriptor_has_attach_ptr_mTlevel3_t_omp_default_mapper
! CHECK: %[[LEVEL3_Z_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(implicit, tofrom, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[LEVEL3_Z_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[LEVEL3_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTlevel3_t{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_mTlevel3_t{{.*}}>) map_clauses(implicit, tofrom) capture(ByRef) members(%[[LEVEL3_Z_PTEE]] : [0] : {{.*}}) name("") -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTlevel3_t{{.*}}>>
! CHECK: omp.declare_mapper.info map_entries(%[[LEVEL3_PARENT]], %[[LEVEL3_Z_PTEE]], %[[LEVEL3_Z_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QMdescriptor_has_attach_ptr_mTlevel2_t_omp_default_mapper
! CHECK: %[[LEVEL2_L3_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} mapper{{.*}}_QMdescriptor_has_attach_ptr_mTlevel3_t_omp_default_mapper{{.*}} name("") has_attach_ptr(true)
! CHECK: %[[LEVEL2_L3_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[LEVEL2_PARENT:.*]] = omp.map.info {{.*}} members(%[[LEVEL2_L3_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[LEVEL2_PARENT]], %[[LEVEL2_L3_PTEE]], %[[LEVEL2_L3_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_mlevel1_t_omp_default_mapper
! CHECK: %[[LEVEL1_L2_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} mapper{{.*}}_QMdescriptor_has_attach_ptr_mTlevel2_t_omp_default_mapper{{.*}} name("") has_attach_ptr(true)
! CHECK: %[[LEVEL1_L2_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[LEVEL1_PARENT:.*]] = omp.map.info {{.*}} members(%[[LEVEL1_L2_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[LEVEL1_PARENT]], %[[LEVEL1_L2_PTEE]], %[[LEVEL1_L2_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QMdescriptor_has_attach_ptr_mTscalar_block_t_omp_default_mapper
! CHECK: %[[SCALAR_BLOCK_ARRAY_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(implicit, tofrom, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[SCALAR_BLOCK_ARRAY_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[SCALAR_BLOCK_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>) map_clauses(implicit, tofrom) capture(ByRef) members(%[[SCALAR_BLOCK_ARRAY_PTEE]] : [1] : {{.*}}) name("") -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>
! CHECK: omp.declare_mapper.info map_entries(%[[SCALAR_BLOCK_PARENT]], %[[SCALAR_BLOCK_ARRAY_PTEE]], %[[SCALAR_BLOCK_ARRAY_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_mcontainer_t_omp_default_mapper
! CHECK: %[[CONTAINER_BLK_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} mapper{{.*}}_QMdescriptor_has_attach_ptr_mTscalar_block_t_omp_default_mapper{{.*}} name("") has_attach_ptr(true)
! CHECK: %[[CONTAINER_BLK_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[CONTAINER_PARENT:.*]] = omp.map.info {{.*}} members(%[[CONTAINER_BLK_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[CONTAINER_PARENT]], %[[CONTAINER_BLK_PTEE]], %[[CONTAINER_BLK_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_minner_t_omp_default_mapper
! CHECK: %[[INNER_QQ_ARRAY_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} name("") has_attach_ptr(true)
! CHECK: %[[INNER_QQ_ARRAY_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[INNER_QQ_PARENT:.*]] = omp.map.info {{.*}} members(%[[INNER_QQ_ARRAY_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[INNER_QQ_PARENT]], %[[INNER_QQ_ARRAY_PTEE]], %[[INNER_QQ_ARRAY_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QMdescriptor_has_attach_ptr_mTinner_t_omp_default_mapper
! CHECK: %[[INNER_T_ARRAY_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} name("") has_attach_ptr(true)
! CHECK: %[[INNER_T_ARRAY_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[INNER_T_PARENT:.*]] = omp.map.info {{.*}} members(%[[INNER_T_ARRAY_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[INNER_T_PARENT]], %[[INNER_T_ARRAY_PTEE]], %[[INNER_T_ARRAY_ATTACH]] : {{.*}})

! CHECK: omp.declare_mapper @_QQMdescriptor_has_attach_ptr_mouter_t_omp_default_mapper
! CHECK: %[[OUTER_COMP_PTEE:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom, ref_ptee) {{.*}} mapper{{.*}}_QMdescriptor_has_attach_ptr_mTinner_t_omp_default_mapper{{.*}} name("") has_attach_ptr(true)
! CHECK: %[[OUTER_COMP_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptee) {{.*}} name("")
! CHECK: %[[OUTER_PARENT:.*]] = omp.map.info {{.*}} members(%[[OUTER_COMP_PTEE]] : [0] : {{.*}}) name("")
! CHECK: omp.declare_mapper.info map_entries(%[[OUTER_PARENT]], %[[OUTER_COMP_PTEE]], %[[OUTER_COMP_ATTACH]] : {{.*}})

contains

subroutine basic_alloc(a)
  type(dtype_a) :: a

!$omp target map(to: a%var_a)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPbasic_alloc
! CHECK: %[[BASIC_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("a%var_a") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[BASIC_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) name("a%var_a") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BASIC_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("a%var_a") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BASIC_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTdtype_a{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_mTdtype_a{{.*}}>) map_clauses(storage) capture(ByRef) members(%[[BASIC_DESC]], %[[BASIC_PTEE]] : [0], [0, 0] : {{.*}}) name("a") partial_map(true) -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTdtype_a{{.*}}>>
! CHECK: omp.target kernel_type(generic) map_entries(%[[BASIC_PARENT]] -> {{.*}}, %[[BASIC_DESC]] -> {{.*}}, %[[BASIC_ATTACH]] -> {{.*}}, %[[BASIC_PTEE]] -> {{.*}} : {{.*}})

end subroutine basic_alloc

subroutine nested_alloc(s)
  type(outer_t) :: s

!$omp target map(tofrom: s)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPnested_alloc
! CHECK: %[[NESTED_TARGET_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} mapper{{.*}}_QQMdescriptor_has_attach_ptr_mouter_t_omp_default_mapper{{.*}} name("s")
! CHECK: omp.target kernel_type(generic) map_entries(%[[NESTED_TARGET_MAP]] -> {{.*}} : {{.*}})

end subroutine nested_alloc

subroutine pointer_object_map(p)
  type(inner_t), pointer :: p

!$omp target map(to: p, p%arr)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPpointer_object_map
! CHECK: %[[PTR_ARR_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("p%arr") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[PTR_ARR_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) name("p%arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[PTR_ARR_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("p%arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[PTR_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>) map_clauses(to) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>) name("p") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>
! CHECK: %[[PTR_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>) map_clauses(always, to) capture(ByRef) members(%[[PTR_PTEE]], %[[PTR_ARR_DESC]], %[[PTR_ARR_PTEE]] : [0], [0, 0], [0, 0, 0] : {{.*}}) name("p") -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>>
! CHECK: %[[PTR_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>>, !fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>) name("p") -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMdescriptor_has_attach_ptr_mTinner_t{{.*}}>>>>
! CHECK: omp.target kernel_type(generic) map_entries(%[[PTR_PARENT]] -> {{.*}}, %[[PTR_ARR_DESC]] -> {{.*}}, %[[PTR_ARR_ATTACH]] -> {{.*}}, %[[PTR_ATTACH]] -> {{.*}}, %[[PTR_PTEE]] -> {{.*}}, %[[PTR_ARR_PTEE]] -> {{.*}} : {{.*}})

end subroutine pointer_object_map

subroutine scalar_and_array_members(c)
  type(container_t) :: c

!$omp target map(tofrom: c)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPscalar_and_array_members
! CHECK: %[[SCALAR_TARGET_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} mapper{{.*}}_QQMdescriptor_has_attach_ptr_mcontainer_t_omp_default_mapper{{.*}} name("c")
! CHECK: omp.target kernel_type(generic) map_entries(%[[SCALAR_TARGET_MAP]] -> {{.*}} : {{.*}})

end subroutine scalar_and_array_members

subroutine explicit_map_all_exp_map(e)
  type(exp_map_t) :: e

!$omp target map(tofrom: e%blk, e%blk%scalar, e%blk%array, e%scalar_f)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPexplicit_map_all_exp_map
! CHECK: %[[EXP_BLK_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>>, !fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>) mapper{{.*}}_QQMdescriptor_has_attach_ptr_mscalar_block_t_omp_default_mapper{{.*}} name("e%blk") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>
! CHECK: %[[EXP_BLK_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>>, !fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>) map_clauses(always, to) capture(ByRef) name("e%blk") -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>>
! CHECK: %[[EXP_BLK_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>>, !fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>, !fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>) name("e%blk") -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdescriptor_has_attach_ptr_mTscalar_block_t{{.*}}>>>>
! CHECK: %[[EXP_BLK_SCALAR:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("e%blk%scalar") has_attach_ptr(true) -> !fir.ref<i32>
! CHECK: %[[EXP_ARRAY_PTEE:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("e%blk%array") has_attach_ptr(true) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
! CHECK: %[[EXP_ARRAY_DESC:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) name("e%blk%array") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[EXP_ARRAY_ATTACH:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) {{.*}} name("e%blk%array") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[EXP_SCALAR_F:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) name("e%scalar_f") -> !fir.ref<f32>
! CHECK: %[[EXP_PARENT:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTexp_map_t{{.*}}>>, !fir.type<_QMdescriptor_has_attach_ptr_mTexp_map_t{{.*}}>) map_clauses(storage) capture(ByRef) members(%[[EXP_BLK_DESC]], %[[EXP_BLK_PTEE]], %[[EXP_BLK_SCALAR]], %[[EXP_ARRAY_DESC]], %[[EXP_ARRAY_PTEE]], %[[EXP_SCALAR_F]] : [0], [0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1, 0], [1] : {{.*}}) name("e") partial_map(true) -> !fir.ref<!fir.type<_QMdescriptor_has_attach_ptr_mTexp_map_t{{.*}}>>
! CHECK: omp.target kernel_type(generic) map_entries(%[[EXP_PARENT]] -> {{.*}}, %[[EXP_BLK_DESC]] -> {{.*}}, %[[EXP_BLK_SCALAR]] -> {{.*}}, %[[EXP_ARRAY_DESC]] -> {{.*}}, %[[EXP_SCALAR_F]] -> {{.*}}, %[[EXP_BLK_ATTACH]] -> {{.*}}, %[[EXP_ARRAY_ATTACH]] -> {{.*}}, %[[EXP_BLK_PTEE]] -> {{.*}}, %[[EXP_ARRAY_PTEE]] -> {{.*}} : {{.*}})

end subroutine explicit_map_all_exp_map

subroutine deeply_nested_levels(v)
  type(level1_t) :: v

!$omp target map(tofrom: v)
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPdeeply_nested_levels
! CHECK: %[[DEEP_TARGET_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} mapper{{.*}}_QQMdescriptor_has_attach_ptr_mlevel1_t_omp_default_mapper{{.*}} name("v")
! CHECK: omp.target kernel_type(generic) map_entries(%[[DEEP_TARGET_MAP]] -> {{.*}} : {{.*}})

end subroutine deeply_nested_levels

subroutine mapper_and_implicit(x)
  type(map_dtype_t) :: x

!$omp declare mapper (map_dtype_t :: v) map(v, v%arr, v%scalar)

!$omp target map(mapper(default): x)
!$omp end target

!$omp target
  x%scalar = x%scalar + 1
!$omp end target

! CHECK-LABEL: func.func @_QMdescriptor_has_attach_ptr_mPmapper_and_implicit
! CHECK: %[[TARGET_MAPPER_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} mapper{{.*}}_QQMdescriptor_has_attach_ptr_mFmapper_and_implicitmap_dtype_t_omp_default_mapper{{.*}} name("x")
! CHECK: omp.target kernel_type(generic) map_entries(%[[TARGET_MAPPER_MAP]] -> {{.*}} : {{.*}})
! CHECK: %[[TARGET_IMPLICIT_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} mapper{{.*}}_QQMdescriptor_has_attach_ptr_mFmapper_and_implicitmap_dtype_t_omp_default_mapper{{.*}} name("x")
! CHECK: omp.target kernel_type(generic) map_entries(%[[TARGET_IMPLICIT_MAP]] -> {{.*}} : {{.*}})

end subroutine mapper_and_implicit

end module descriptor_has_attach_ptr_m
