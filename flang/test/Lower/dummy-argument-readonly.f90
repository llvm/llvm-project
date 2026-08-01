! Test the fir.read_only marker produced by CallInterface for the supported
! subset of INTENT(IN) dummy data objects. The marker is emitted during
! lowering independently of the optimization level.
!
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine scalar_intent_in(x)
  integer, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPscalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only}) {

subroutine real_scalar_intent_in(x)
  real, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPreal_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<f32> {fir.bindc_name = "x", fir.read_only}) {

subroutine complex_scalar_intent_in(x)
  complex, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<complex<f32>> {fir.bindc_name = "x", fir.read_only}) {

subroutine logical_scalar_intent_in(x)
  logical, intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPlogical_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x", fir.read_only}) {

recursive subroutine recursive_scalar_intent_in(x)
  integer, intent(in) :: x
  if (x > 0) call recursive_scalar_intent_in(x - 1)
end subroutine
! CHECK-LABEL: func.func @_QPrecursive_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only}) attributes

subroutine optional_scalar_intent_in(x)
  integer, intent(in), optional :: x
end subroutine
! CHECK-LABEL: func.func @_QPoptional_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.optional, fir.read_only}) {

subroutine target_scalar_intent_in(x)
  integer, intent(in), target :: x
end subroutine
! CHECK-LABEL: func.func @_QPtarget_scalar_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only, fir.target}) {

subroutine bindc_definition_intent_in(x) bind(c)
  use iso_c_binding, only : c_int
  integer(c_int), intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @bindc_definition_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only}) attributes

! Arrays are excluded because a forwarded dummy without an INTENT contract may
! require compiler-generated copy-out.
subroutine explicit_shape_intent_in(x)
  integer, intent(in) :: x(10)
end subroutine
! CHECK-LABEL: func.func @_QPexplicit_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x"}) {

subroutine assumed_size_intent_in(x)
  integer, intent(in) :: x(*)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_size_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "x"}) {

subroutine target_explicit_shape_intent_in(x)
  integer, intent(in), target :: x(10)
end subroutine
! CHECK-LABEL: func.func @_QPtarget_explicit_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x", fir.target}) {

subroutine assumed_shape_intent_in(x)
  integer, intent(in) :: x(:)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_shape_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}) {

subroutine assumed_rank_intent_in(x)
  integer, intent(in) :: x(..)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_rank_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.box<!fir.array<*:i32>> {fir.bindc_name = "x"}) {

! CHARACTER and derived types are outside the current conservative subset.
subroutine character_intent_in(x)
  character(len=*), intent(in) :: x
end subroutine
! CHECK-LABEL: func.func @_QPcharacter_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.boxchar<1> {fir.bindc_name = "x"}) {

module readonly_derived_types
  type :: type_with_array
    real :: values(4)
  end type
contains
  subroutine derived_with_array_intent_in(x)
    type(type_with_array), intent(in) :: x
  end subroutine
! CHECK-LABEL: func.func @_QMreadonly_derived_typesPderived_with_array_intent_in(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.type<{{.*}}>> {fir.bindc_name = "x"}) {
end module

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
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x", fir.read_only}) {

subroutine intent_in_allocatable(x)
  integer, intent(in), allocatable :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_allocatable(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x", fir.read_only}) {

subroutine intent_in_pointer_array(x)
  integer, intent(in), pointer :: x(:)
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_pointer_array(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "x", fir.read_only}) {

subroutine intent_in_allocatable_array(x)
  integer, intent(in), allocatable :: x(:)
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_allocatable_array(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x", fir.read_only}) {

! The marker is shallow for descriptor arguments. Defining a POINTER target is
! valid even though the descriptor itself has INTENT(IN).
subroutine intent_in_pointer_target_write(x)
  integer, intent(in), pointer :: x
  x = 42
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_pointer_target_write(
! CHECK-SAME:    %{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x", fir.read_only}) {
! CHECK:         hlfir.assign {{.*}} to {{.*}} : i32, !fir.ptr<i32>

subroutine intent_in_asynchronous(x)
  integer, intent(in), asynchronous :: x
end subroutine
! CHECK-LABEL: func.func @_QPintent_in_asynchronous(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.asynchronous, fir.bindc_name = "x"}) {

! VOLATILE with INTENT(IN) is prohibited by C870 (Fortran 2023) and is already
! covered by test/Semantics/misc-declarations.f90, so it cannot be exercised by
! a valid lowering test. The Volatile exclusion in
! dummyArgCanUseLLVMReadonly remains a defensive check.
