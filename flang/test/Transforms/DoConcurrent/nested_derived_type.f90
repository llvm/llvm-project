! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

! CHECK-DAG: omp.declare_mapper @[[INNER_MAPPER:.*inner_t_omp_default_mapper]] : !fir.type<{{.*}}inner_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[INNER_ALLOC_MAPPER:.*inner_alloc_t_omp_default_mapper]] : !fir.type<{{.*}}inner_alloc_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[INNER_PTR_MAPPER:.*inner_ptr_t_omp_default_mapper]] : !fir.type<{{.*}}inner_ptr_t{{.*}}>
! Character length information must be preserved in the nested mapper type.
! CHECK-DAG: omp.declare_mapper @[[INNER_CHAR_MAPPER:.*inner_char_t_omp_default_mapper]] : !fir.type<{{.*}}inner_char_t{x:i32,name:!fir.char<1,8>}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_MAPPER:.*outer_t_omp_default_mapper]] : !fir.type<{{.*}}outer_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_ARRAY_MAPPER:.*outer_array_t_omp_default_mapper]] : !fir.type<{{.*}}outer_array_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_ALLOC_MAPPER:.*outer_alloc_t_omp_default_mapper]] : !fir.type<{{.*}}outer_alloc_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_PTR_MAPPER:.*outer_ptr_t_omp_default_mapper]] : !fir.type<{{.*}}outer_ptr_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_CHAR_MAPPER:.*outer_char_t_omp_default_mapper]] : !fir.type<{{.*}}outer_char_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[LEVEL2_MAPPER:.*level2_t_omp_default_mapper]] : !fir.type<{{.*}}level2_t{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[LEVEL3_MAPPER:.*level3_t_omp_default_mapper]] : !fir.type<{{.*}}level3_t{{.*}}>
!
! Each outer mapper body must map its component with the matching nested mapper.
! The var_ptr type ties the mapper reference to the correct member map, so a
! nested mapper cannot be satisfied by an unrelated mapper body.
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}inner_t{{.*}}>>, !fir.type<{{.*}}inner_t{{.*}}>) {{.*}}mapper(@[[INNER_MAPPER]]){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.array<2x!fir.type<{{.*}}inner_t{{.*}}>>>, !fir.array<2x!fir.type<{{.*}}inner_t{{.*}}>>) {{.*}}mapper(@[[INNER_MAPPER]]){{.*}}bounds
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}inner_alloc_t{{.*}}>>, !fir.type<{{.*}}inner_alloc_t{{.*}}>) {{.*}}mapper(@[[INNER_ALLOC_MAPPER]]){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}inner_ptr_t{{.*}}>>, !fir.type<{{.*}}inner_ptr_t{{.*}}>) {{.*}}mapper(@[[INNER_PTR_MAPPER]]){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}inner_char_t{{.*}}>>, !fir.type<{{.*}}inner_char_t{{.*}}>) {{.*}}mapper(@[[INNER_CHAR_MAPPER]]){{.*}}name("")
! The three-level record nests mappers all the way down (level3 -> level2 -> inner).
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}level2_t{{.*}}>>, !fir.type<{{.*}}level2_t{{.*}}>) {{.*}}mapper(@[[LEVEL2_MAPPER]]){{.*}}name("")
! Descriptor-based members deep-copy their pointee (ref_ptee) and emit an attach
! entry inside the mapper body. Each is anchored to its descriptor type, so the
! check proves mapper-body correctness, not just that some ref_ptee exists.
! Allocatable array component (inner_alloc_t):
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.box<!fir.heap<!fir.array<?xi32>>>{{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.box<!fir.heap<!fir.array<?xi32>>>{{.*}}) map_clauses(attach, ref_ptee)
! Deferred-length character component (inner_defchar_t): length descriptor preserved.
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.box<!fir.heap<!fir.char<1,?>>>{{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.box<!fir.heap<!fir.char<1,?>>>{{.*}}) map_clauses(attach, ref_ptee)
! Polymorphic class component (outer_poly_t): class descriptor preserved.
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.class<{{.*}}poly_base_t{{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! CHECK-DAG: omp.map.info var_ptr({{.*}}!fir.class<{{.*}}poly_base_t{{.*}}) map_clauses(attach, ref_ptee)
!
! Structural check: each parent mapper body places its component at member index
! [0] and ties that member map to the correct nested type (not just existence).
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}inner_t{{.*}}){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_array_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}inner_t{{.*}}){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_alloc_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}inner_alloc_t{{.*}}){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_ptr_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}inner_ptr_t{{.*}}){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_char_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}inner_char_t{{.*}}){{.*}}name("")
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}level3_t{{.*}}>>, {{.*}}members({{.*}} : [0] : {{.*}}level2_t{{.*}}){{.*}}name("")
!
! A pointer-to-record component is attach-ineligible: a (trivial) mapper is still
! generated for the enclosing type, but its body maps no members (empty member
! list), because the pointer member is skipped and `id` is a trivial scalar. The
! `[^[]` guard proves there is no member-placement index (i.e. no member map).
! CHECK-DAG: omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<{{.*}}outer_ptr_to_rec_t{{.*}}>>, !fir.type<{{.*}}outer_ptr_to_rec_t{{.*}}>) {{.*}}members({{[^[]*}}) name("")
!
! A deferred-length character component is descriptor-based; its length parameter
! is preserved in the nested mapper type via the character box.
! CHECK-DAG: omp.declare_mapper @[[INNER_DEFCHAR_MAPPER:.*inner_defchar_t_omp_default_mapper]] : !fir.type<{{.*}}inner_defchar_t{{.*}}!fir.char<1,?>{{.*}}>
! CHECK-DAG: omp.declare_mapper @[[OUTER_DEFCHAR_MAPPER:.*outer_defchar_t_omp_default_mapper]] : !fir.type<{{.*}}outer_defchar_t{{.*}}>
!
! A polymorphic (class) component keeps its class descriptor in the mapper type.
! CHECK-DAG: omp.declare_mapper @[[OUTER_POLY_MAPPER:.*outer_poly_t_omp_default_mapper]] : !fir.type<{{.*}}outer_poly_t{{.*}}!fir.class<{{.*}}poly_base_t{{.*}}>{{.*}}>
!
! Same-spelled derived types declared in DIFFERENT procedure scopes get distinct,
! scope-qualified default mapper names (no collision). The canonical mapper name
! carries the _QF<proc> scope, so no mangler callback is required in this pass.
! CHECK-DAG: omp.declare_mapper @{{.*}}scope_aT{{.*}}scoped_t_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @{{.*}}scope_bT{{.*}}scoped_t_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @{{.*}}scope_aT{{.*}}leaf_t_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @{{.*}}scope_bT{{.*}}leaf_t_omp_default_mapper

module nested_derived_type_mod
  implicit none

  type :: inner_t
    integer :: x
  end type

  type :: outer_t
    type(inner_t) :: member
  end type

  type :: outer_array_t
    type(inner_t) :: member(2)
  end type

  type :: inner_alloc_t
    integer, allocatable :: values(:)
  end type

  type :: outer_alloc_t
    type(inner_alloc_t) :: member
  end type

  type :: level2_t
    type(inner_t) :: member
  end type

  type :: level3_t
    type(level2_t) :: member
  end type

  type :: inner_ptr_t
    integer :: x
    integer, pointer :: p => null()
  end type

  type :: outer_ptr_t
    type(inner_ptr_t) :: member
  end type

  type :: outer_ptr_to_rec_t
    type(inner_t), pointer :: member => null()
    integer :: id
  end type

  type :: inner_char_t
    integer :: x
    character(len=8) :: name
  end type

  type :: outer_char_t
    type(inner_char_t) :: member
  end type

  type :: inner_defchar_t
    integer :: x
    character(len=:), allocatable :: name
  end type

  type :: outer_defchar_t
    type(inner_defchar_t) :: member
  end type

  type :: poly_base_t
    integer :: x
  end type

  type :: outer_poly_t
    class(poly_base_t), allocatable :: member
  end type
end module

subroutine scalar_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPscalar_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

subroutine array_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_array_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QParray_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_ARRAY_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member(2)%x = i
  end do
end subroutine

subroutine allocatable_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_alloc_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPallocatable_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_ALLOC_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  ! Whole-array allocatable assignment; this test only checks that the captured
  ! variable gets the outer mapper. Allocation/reallocation lowering is incidental here.
  do concurrent (i = 1:4)
    a(i)%member%values = [i]
  end do
end subroutine

subroutine deep_nested()
  use nested_derived_type_mod
  implicit none
  type(level3_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPdeep_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[LEVEL3_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%member%x = i
  end do
end subroutine

subroutine pointer_field_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_ptr_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPpointer_field_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_PTR_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

subroutine char_component_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_char_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPchar_component_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_CHAR_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

subroutine pointer_to_record_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_ptr_to_rec_t) :: a(4)
  integer :: i

  ! A pointer-to-record component is attach-ineligible. A (trivial) mapper is
  ! still generated for the enclosing type and attached to the captured variable,
  ! but its mapper body maps no members (verified above via the empty member list).
  ! CHECK-LABEL: func.func @_QPpointer_to_record_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@{{.*}}outer_ptr_to_rec_t_omp_default_mapper) {{.*}}name("_QFpointer_to_record_nestedEa")
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%id = i
  end do
end subroutine

! The following two subroutines each declare LOCAL derived types with the SAME
! source names (leaf_t/scoped_t) but different structure. They exercise mapper
! symbol-name uniqueness across scopes: each scope gets its own scope-qualified
! mapper (checked in the CHECK-DAG block near the top of the file).
subroutine scope_a()
  implicit none
  type :: leaf_t
    integer :: x
  end type
  type :: scoped_t
    type(leaf_t) :: member
  end type
  type(scoped_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPscope_a()
  ! CHECK: omp.map.info {{.*}} mapper(@{{.*}}scope_aT{{.*}}scoped_t_omp_default_mapper) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

subroutine scope_b()
  implicit none
  type :: leaf_t
    real :: y
  end type
  type :: scoped_t
    type(leaf_t) :: member
  end type
  type(scoped_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPscope_b()
  ! CHECK: omp.map.info {{.*}} mapper(@{{.*}}scope_bT{{.*}}scoped_t_omp_default_mapper) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%y = real(i)
  end do
end subroutine

! A nested record with a deferred-length character component still lowers; the
! character length is carried in the descriptor inside the mapper.
subroutine defchar_component_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_defchar_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPdefchar_component_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_DEFCHAR_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

! A nested record with a polymorphic (class) component still lowers; the class
! descriptor is preserved in the mapper.
subroutine poly_component_nested()
  use nested_derived_type_mod
  implicit none
  type(outer_poly_t) :: a(4)
  integer :: i

  ! CHECK-LABEL: func.func @_QPpoly_component_nested()
  ! CHECK: omp.map.info {{.*}} mapper(@[[OUTER_POLY_MAPPER]]) bounds
  ! CHECK: omp.target kernel_type(spmd)
  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine
