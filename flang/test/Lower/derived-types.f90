! Test basic parts of derived type entities lowering
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Note: only testing non parameterized derived type here.

module d
  type r
    real :: x
  end type
  type r2
    real :: x_array(10, 20)
  end type
  type c
    character(10) :: ch
  end type
  type c2
    character(10) :: ch_array(20, 30)
  end type
contains

! -----------------------------------------------------------------------------
!            Test simple derived type symbol lowering 
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMdPderived_dummy(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMdTr{x:f32}>>{{.*}}, %{{.*}}: !fir.ref<!fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>>{{.*}}) {
subroutine derived_dummy(some_r, some_c2)
  type(r) :: some_r
  type(c2) :: some_c2
end subroutine

! CHECK-LABEL: func @_QMdPlocal_derived(
subroutine local_derived()
  ! CHECK-DAG: fir.alloca !fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>
  ! CHECK-DAG: fir.alloca !fir.type<_QMdTr{x:f32}>
  type(r) :: some_r
  type(c2) :: some_c2
end subroutine

! CHECK-LABEL: func @_QMdPsaved_derived(
subroutine saved_derived()
  ! CHECK-DAG: fir.address_of(@_QMdFsaved_derivedEsome_c2) : !fir.ref<!fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>>
  ! CHECK-DAG: fir.address_of(@_QMdFsaved_derivedEsome_r) : !fir.ref<!fir.type<_QMdTr{x:f32}>>
  type(r), save :: some_r
  type(c2), save :: some_c2
  call use_symbols(some_r, some_c2)
end subroutine


! -----------------------------------------------------------------------------
!            Test simple derived type references 
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMdPscalar_numeric_ref(
subroutine scalar_numeric_ref()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.type<_QMdTr{x:f32}>
  type(r) :: some_r
  ! CHECK: %[[field:.*]] = fir.field_index x, !fir.type<_QMdTr{x:f32}>
  ! CHECK: fir.coordinate_of %[[alloc]], %[[field]] : (!fir.ref<!fir.type<_QMdTr{x:f32}>>, !fir.field) -> !fir.ref<f32>
  call real_bar(some_r%x)
end subroutine

! CHECK-LABEL: func @_QMdPscalar_character_ref(
subroutine scalar_character_ref()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.type<_QMdTc{ch:!fir.char<1,10>}>
  type(c) :: some_c
  ! CHECK: %[[field:.*]] = fir.field_index ch, !fir.type<_QMdTc{ch:!fir.char<1,10>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[alloc]], %[[field]] : (!fir.ref<!fir.type<_QMdTc{ch:!fir.char<1,10>}>>, !fir.field) -> !fir.ref<!fir.char<1,10>>
  ! CHECK-DAG: %[[c10:.*]] = arith.constant 10 : index
  ! CHECK-DAG: %[[conv:.*]] = fir.convert %[[coor]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: fir.emboxchar %[[conv]], %c10 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  call char_bar(some_c%ch)
end subroutine

! FIXME: coordinate of generated for derived%array_comp(i) are not zero based as they
! should be.

! CHECK-LABEL: func @_QMdParray_comp_elt_ref(
subroutine array_comp_elt_ref()
  type(r2) :: some_r2
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.type<_QMdTr2{x_array:!fir.array<10x20xf32>}>
  ! CHECK: %[[field:.*]] = fir.field_index x_array, !fir.type<_QMdTr2{x_array:!fir.array<10x20xf32>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[alloc]], %[[field]] : (!fir.ref<!fir.type<_QMdTr2{x_array:!fir.array<10x20xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10x20xf32>>
  ! CHECK-DAG: %[[index1:.*]] = arith.subi %c5{{.*}}, %c1{{.*}} : i64
  ! CHECK-DAG: %[[index2:.*]] = arith.subi %c6{{.*}}, %c1{{.*}} : i64
  ! CHECK: fir.coordinate_of %[[coor]], %[[index1]], %[[index2]] : (!fir.ref<!fir.array<10x20xf32>>, i64, i64) -> !fir.ref<f32>
  call real_bar(some_r2%x_array(5, 6))
end subroutine


! CHECK-LABEL: func @_QMdPchar_array_comp_elt_ref(
subroutine char_array_comp_elt_ref()
  type(c2) :: some_c2
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>>, !fir.field) -> !fir.ref<!fir.array<20x30x!fir.char<1,10>>>
  ! CHECK-DAG: %[[index1:.*]] = arith.subi %c5{{.*}}, %c1{{.*}} : i64
  ! CHECK-DAG: %[[index2:.*]] = arith.subi %c6{{.*}}, %c1{{.*}} : i64
  ! CHECK: fir.coordinate_of %[[coor]], %[[index1]], %[[index2]] : (!fir.ref<!fir.array<20x30x!fir.char<1,10>>>, i64, i64) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: fir.emboxchar %{{.*}}, %c10 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  call char_bar(some_c2%ch_array(5, 6))
end subroutine

! CHECK: @_QMdParray_elt_comp_ref
subroutine array_elt_comp_ref()
  type(r) :: some_r_array(100)
  ! CHECK: %[[alloca:.*]] = fir.alloca !fir.array<100x!fir.type<_QMdTr{x:f32}>>
  ! CHECK: %[[index:.*]] = arith.subi %c5{{.*}}, %c1{{.*}} : i64
  ! CHECK: %[[elt:.*]] = fir.coordinate_of %[[alloca]], %[[index]] : (!fir.ref<!fir.array<100x!fir.type<_QMdTr{x:f32}>>>, i64) -> !fir.ref<!fir.type<_QMdTr{x:f32}>>
  ! CHECK: %[[field:.*]] = fir.field_index x, !fir.type<_QMdTr{x:f32}>
  ! CHECK: fir.coordinate_of %[[elt]], %[[field]] : (!fir.ref<!fir.type<_QMdTr{x:f32}>>, !fir.field) -> !fir.ref<f32>
  call real_bar(some_r_array(5)%x)
end subroutine

! CHECK: @_QMdPchar_array_elt_comp_ref
subroutine char_array_elt_comp_ref()
  type(c) :: some_c_array(100)
  ! CHECK: fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMdTc{ch:!fir.char<1,10>}>>>, i64) -> !fir.ref<!fir.type<_QMdTc{ch:!fir.char<1,10>}>>
  ! CHECK: fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.type<_QMdTc{ch:!fir.char<1,10>}>>, !fir.field) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: fir.emboxchar %{{.*}}, %c10{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  call char_bar(some_c_array(5)%ch)
end subroutine

! -----------------------------------------------------------------------------
!            Test loading derived type components
! -----------------------------------------------------------------------------

! Most of the other tests only require lowering code to compute the address of
! components. This one requires loading a component which tests other code paths
! in lowering.

! CHECK-LABEL: func @_QMdPscalar_numeric_load(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QMdTr{x:f32}>>
real function scalar_numeric_load(some_r)
  type(r) :: some_r
  ! CHECK: %[[field:.*]] = fir.field_index x, !fir.type<_QMdTr{x:f32}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg0]], %[[field]] : (!fir.ref<!fir.type<_QMdTr{x:f32}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK: fir.load %[[coor]]
  scalar_numeric_load = some_r%x
end function

! -----------------------------------------------------------------------------
!            Test returned derived types (no length parameters)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMdPbar_return_derived() -> !fir.type<_QMdTr{x:f32}>
function bar_return_derived()
  ! CHECK: %[[res:.*]] = fir.alloca !fir.type<_QMdTr{x:f32}>
  type(r) :: bar_return_derived
  ! CHECK: %[[resLoad:.*]] = fir.load %[[res]] : !fir.ref<!fir.type<_QMdTr{x:f32}>>
  ! CHECK: return %[[resLoad]] : !fir.type<_QMdTr{x:f32}>
end function

! CHECK-LABEL: func @_QMdPcall_bar_return_derived(
subroutine call_bar_return_derived()
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMdTr{x:f32}>
  ! CHECK: %[[call:.*]] = fir.call @_QMdPbar_return_derived() : () -> !fir.type<_QMdTr{x:f32}>
  ! CHECK: fir.save_result %[[call]] to %[[tmp]] : !fir.type<_QMdTr{x:f32}>, !fir.ref<!fir.type<_QMdTr{x:f32}>>
  ! CHECK: fir.call @_QPr_bar(%[[tmp]]) : (!fir.ref<!fir.type<_QMdTr{x:f32}>>) -> ()
  call r_bar(bar_return_derived())
end subroutine

end module

! -----------------------------------------------------------------------------
!            Test derived type with pointer/allocatable components 
! -----------------------------------------------------------------------------

module d2
  type recursive_t
    real :: x
    type(recursive_t), pointer :: ptr
  end type
contains
! CHECK-LABEL: func @_QMd2Ptest_recursive_type(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMd2Trecursive_t{x:f32,ptr:!fir.box<!fir.ptr<!fir.type<_QMd2Trecursive_t>>>}>>{{.*}}) {
subroutine test_recursive_type(some_recursive)
  type(recursive_t) :: some_recursive
end subroutine
end module

! -----------------------------------------------------------------------------
!            Test global derived type symbol lowering 
! -----------------------------------------------------------------------------

module data_mod
  use d
  type(r) :: some_r
  type(c2) :: some_c2
end module

! Test globals

! CHECK-DAG: fir.global @_QMdata_modEsome_c2 : !fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>
! CHECK-DAG: fir.global @_QMdata_modEsome_r : !fir.type<_QMdTr{x:f32}>
! CHECK-DAG: fir.global internal @_QMdFsaved_derivedEsome_c2 : !fir.type<_QMdTc2{ch_array:!fir.array<20x30x!fir.char<1,10>>}>
! CHECK-DAG: fir.global internal @_QMdFsaved_derivedEsome_r : !fir.type<_QMdTr{x:f32}>
