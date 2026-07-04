! Test the fir.read_only marker produced by CallInterface for INTENT(IN) dummy
! data objects. The marker is emitted during lowering independently of the
! optimization level.
!
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine scalar_intent_in(x)
  integer, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only}) {

subroutine optional_scalar_intent_in(x)
  integer, intent(in), optional :: x
end subroutine
! CHECK-LABEL: func.func @_QPoptional_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.optional, fir.read_only}) {

subroutine explicit_shape_intent_in(x)
  integer, intent(in) :: x(10)
end subroutine
! CHECK-LABEL: func.func @_QPexplicit_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x", fir.read_only}) {

subroutine assumed_size_intent_in(x)
  integer, intent(in) :: x(*)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_size_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "x", fir.read_only}) {

subroutine target_scalar_intent_in(x)
  integer, intent(in), target :: x
end subroutine
! CHECK-LABEL: func.func @_QPtarget_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only, fir.target}) {

subroutine target_explicit_shape_intent_in(x)
  integer, intent(in), target :: x(10)
end subroutine
! CHECK-LABEL: func.func @_QPtarget_explicit_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x", fir.read_only, fir.target}) {

! CHARACTER uses the dedicated boxchar path in handleImplicitDummy. The
! current optimization does not annotate the data pointer inside boxchar.
subroutine character_intent_in(x)
  character(len=*), intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPcharacter_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.boxchar<1> {fir.bindc_name = "x"}) {

module readonly_derived_types
  type :: plain_type
    integer :: value
  end type
  type :: allocatable_component_type
    integer, allocatable :: values(:)
  end type
  type :: pointer_component_type
    integer, pointer :: value
  end type
contains
  subroutine derived_plain_intent_in(x)
    type(plain_type), intent(in) :: x
  end subroutine
! CHECK-LABEL: func.func @_QMreadonly_derived_typesPderived_plain_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.type<{{.*}}>> {fir.bindc_name = "x", fir.read_only}) {

  subroutine derived_allocatable_intent_in(x)
    type(allocatable_component_type), intent(in) :: x
  end subroutine
! CHECK-LABEL: func.func @_QMreadonly_derived_typesPderived_allocatable_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.type<{{.*}}>> {fir.bindc_name = "x", fir.read_only}) {

  subroutine derived_pointer_intent_in(x)
    type(pointer_component_type), intent(in) :: x
    ! Defining the target through the loaded pointer component is permitted and
    ! does not violate the shallow LLVM readonly contract on x.
    if (associated(x%value)) x%value = 42
  end subroutine
! CHECK-LABEL: func.func @_QMreadonly_derived_typesPderived_pointer_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.type<{{.*}}>> {fir.bindc_name = "x", fir.read_only}) {
end module

subroutine bindc_definition_intent_in(x) bind(c)
  use iso_c_binding, only : c_int
  integer(c_int), intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @bindc_definition_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only}) attributes

! Assumed-shape arguments are boxes. CallInterface records the source contract,
! but FunctionAttr currently translates the marker only on ReferenceType.
subroutine assumed_shape_intent_in(x)
  integer, intent(in) :: x(:)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x", fir.read_only}) {

subroutine intent_inout(x)
  integer, intent(inout) :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_inout(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x"}) {

subroutine intent_out(x)
  integer, intent(out) :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_out(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x"}) {

subroutine intent_unspecified(x)
  integer :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_unspecified(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x"}) {

subroutine intent_in_value(x)
  integer, intent(in), value :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_value(
! CHECK-SAME:    %{{.*}}: i32 {fir.bindc_name = "x"}) {

subroutine intent_in_pointer(x)
  integer, intent(in), pointer :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_pointer(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x"}) {

subroutine intent_in_allocatable(x)
  integer, intent(in), allocatable :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_allocatable(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x"}) {

subroutine intent_in_asynchronous(x)
  integer, intent(in), asynchronous :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_asynchronous(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.asynchronous, fir.bindc_name = "x"}) {

! VOLATILE with INTENT(IN) is prohibited by C870 (Fortran 2023) and is already
! covered by test/Semantics/misc-declarations.f90, so it cannot be exercised by
! a valid lowering test. The Volatile exclusion in dummyArgIsReadOnly remains a
! defensive check.
