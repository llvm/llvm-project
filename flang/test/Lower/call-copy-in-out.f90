! Test copy-in / copy-out of non-contiguous variable passed as F77 array arguments.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Nominal test
! CHECK-LABEL: func @_QPtest_assumed_shape_to_array(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_assumed_shape_to_array(x)
  real :: x(:)
! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %[[x]]
! CHECK: %[[copy_in:.*]]:2 = hlfir.copy_in %[[x_decl]]#0 {{.*}}
! CHECK: %[[addr:.*]] = fir.box_addr %[[copy_in]]#0
! CHECK: fir.call @_QPbar(%[[addr]])
! CHECK: hlfir.copy_out %{{.*}}, %[[copy_in]]#1 to %[[x_decl]]#0
  call bar(x)
end subroutine

! Test that copy-in/copy-out does not trigger the re-evaluation of
! the designator expression.
! CHECK-LABEL: func @_QPeval_expr_only_once(
subroutine eval_expr_only_once(x)
  integer :: only_once
  real :: x(200)
! CHECK: fir.call @_QPonly_once()
! CHECK: hlfir.designate
! CHECK: hlfir.copy_in
! CHECK-NOT: fir.call @_QPonly_once()
! CHECK: fir.call @_QPbar
! CHECK-NOT: fir.call @_QPonly_once()
! CHECK: hlfir.copy_out
! CHECK-NOT: fir.call @_QPonly_once()
  call bar(x(1:200:only_once()))
end subroutine

! Test no copy-in/copy-out is generated for contiguous assumed shapes.
! CHECK-LABEL: func @_QPtest_contiguous(
subroutine test_contiguous(x)
  real, contiguous :: x(:)
! CHECK-NOT: hlfir.copy_in
! CHECK: fir.call @_QPbar
! CHECK-NOT: hlfir.copy_out
  call bar(x)
end subroutine

! Test the parenthesis are preventing copy-out.
! CHECK-LABEL: func @_QPtest_parenthesis(
subroutine test_parenthesis(x)
  real :: x(:)
! CHECK: hlfir.elemental
! CHECK: hlfir.associate
! CHECK: fir.call @_QPbar
! CHECK: hlfir.end_associate
! CHECK-NOT: hlfir.copy_out
  call bar((x))
end subroutine

! Test copy-in in is skipped for intent(out) arguments.
! CHECK-LABEL: func @_QPtest_intent_out(
subroutine test_intent_out(x)
  real :: x(:)
  interface
  subroutine bar_intent_out(x)
    real, intent(out) :: x(100)
  end subroutine
  end interface
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_intent_out
! CHECK: hlfir.copy_out
  call bar_intent_out(x)
end subroutine

! Test copy-out is skipped for intent(out) arguments.
! CHECK-LABEL: func.func @_QPtest_intent_in(
subroutine test_intent_in(x)
  real :: x(:)
  interface
  subroutine bar_intent_in(x)
    real, intent(in) :: x(100)
  end subroutine
  end interface
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_intent_in
! CHECK: hlfir.copy_out
! CHECK-SAME: ) -> ()
  call bar_intent_in(x)
end subroutine

! Test copy-in/copy-out is done for intent(inout)
! CHECK-LABEL: func @_QPtest_intent_inout(
subroutine test_intent_inout(x)
  real :: x(:)
  interface
  subroutine bar_intent_inout(x)
    real, intent(inout) :: x(100)
  end subroutine
  end interface
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_intent_inout
! CHECK: hlfir.copy_out
! CHECK-SAME: to
  call bar_intent_inout(x)
end subroutine

! Test characters are handled correctly
! CHECK-LABEL: func @_QPtest_char(
subroutine test_char(x)
  character(10) :: x(:)
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_char
! CHECK: hlfir.copy_out
  call bar_char(x)
end subroutine test_char

! CHECK-LABEL: func @_QPtest_scalar_substring_does_no_trigger_copy_inout
subroutine test_scalar_substring_does_no_trigger_copy_inout(c, i, j)
  character(*) :: c
  integer :: i, j
  ! CHECK: hlfir.designate
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_char_2
  ! CHECK-NOT: hlfir.copy_out
  call bar_char_2(c(i:j))
end subroutine

! CHECK-LABEL: func @_QPissue871(
subroutine issue871(p)
  ! Test passing implicit derived from scalar pointer (no copy-in/out).
  type t
    integer :: i
  end type t
  type(t), pointer :: p
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_derived
  call bar_derived(p)
end subroutine

! CHECK-LABEL: func @_QPissue871_array(
subroutine issue871_array(p)
  ! Test passing implicit derived from contiguous pointer (no copy-in/out).
  type t
    integer :: i
  end type t
  type(t), pointer, contiguous :: p(:)
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_derived_array
  call bar_derived_array(p)
end subroutine

! CHECK-LABEL: func @_QPwhole_components()
subroutine whole_components()
  ! Test no copy is made for whole components.
  type t
    integer :: i(100)
  end type
  type(t) :: a
  ! CHECK: hlfir.designate
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_integer
  call bar_integer(a%i)
end subroutine

! CHECK-LABEL: func @_QPwhole_component_contiguous_pointer()
subroutine whole_component_contiguous_pointer()
  ! Test no copy is made for whole contiguous pointer components.
  type t
    integer, pointer, contiguous :: i(:)
  end type
  type(t) :: a
  ! CHECK: hlfir.designate
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_integer
  call bar_integer(a%i)
end subroutine

! CHECK-LABEL: func @_QPwhole_component_contiguous_char_pointer()
subroutine whole_component_contiguous_char_pointer()
  ! Test no copy is made for whole contiguous character pointer components.
  type t
    character(:), pointer, contiguous :: i(:)
  end type
  type(t) :: a
  ! CHECK: hlfir.designate
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_char_3
  call bar_char_3(a%i)
end subroutine
