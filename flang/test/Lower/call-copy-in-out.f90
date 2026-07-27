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
! CHECK-SAME: %{{.*}}, %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1) -> ()
  call bar_intent_in(x)
end subroutine

! Test copy-out is skipped when the actual argument has INTENT(IN).
! A copy-in is still needed for the contiguous-requiring callee, but
! the temp is only deallocated on return, not copied back (no "to" clause).
! CHECK-LABEL: func @_QPtest_actual_arg_intent_in(
subroutine test_actual_arg_intent_in(x)
  real, intent(in) :: x(:)
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar
! CHECK: hlfir.copy_out
! CHECK-NOT: to
! CHECK: return
  call bar(x)
end subroutine

! Test the transitive copy-out case. The outer INTENT(IN) array is forwarded
! to a dummy without INTENT, which then passes a noncontiguous section to an
! implicit-interface procedure. Lowering cannot recognize the outer contract
! at the inner call site, so the inner procedure still generates copy-back.
! The outer array must therefore not be marked fir.read_only.
! CHECK-LABEL: func.func @_QPtest_forwarded_intent_in(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<4xf32>> {fir.bindc_name = "x"}) {
subroutine test_forwarded_intent_in(x)
  real, intent(in) :: x(4)
  call test_forwarded_without_intent(x)
end subroutine

! CHECK-LABEL: func.func @_QPtest_forwarded_without_intent(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<4xf32>> {fir.bindc_name = "x"}) {
subroutine test_forwarded_without_intent(x)
  real :: x(4)
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar
! CHECK: hlfir.copy_out
! CHECK-SAME: to
  call bar(x(1:4:2))
end subroutine

! Test copy-out is NOT skipped when passing a section of a pointer component
! of an INTENT(IN) dummy: the pointer target is not a subobject of the dummy
! (F2023 9.4.2 p5), so the callee may define it and copy-out is required.
! CHECK-LABEL: func @_QPtest_actual_arg_intent_in_ptr_component(
subroutine test_actual_arg_intent_in_ptr_component(x)
  type :: t
    integer, pointer :: p(:)
  end type
  type(t), intent(in) :: x
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_integer
! CHECK: hlfir.copy_out
! CHECK-SAME: to
  call bar_integer(x%p(1:4:2))
end subroutine

! Test copy-out is NOT skipped when the actual is a section of an INTENT(IN)
! pointer dummy: INTENT(IN) on a pointer restricts pointer association, not
! the target's contents, so the callee may define the target and copy-out is
! required.
! CHECK-LABEL: func.func @_QPtest_actual_intent_in_pointer(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "pi", fir.read_only}) {
subroutine test_actual_intent_in_pointer(pi)
  integer, intent(in), pointer :: pi(:)
! CHECK: hlfir.copy_in
! CHECK: fir.call @_QPbar_integer
! CHECK: hlfir.copy_out
! CHECK-SAME: to
  call bar_integer(pi(1:4:2))
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

! CHECK-LABEL: func @_QPderived_pointer_no_copy(
subroutine derived_pointer_no_copy(p)
  ! Test passing implicit derived from scalar pointer (no copy-in/out).
  type t
    integer :: i
  end type t
  type(t), pointer :: p
  ! CHECK-NOT: hlfir.copy_in
  ! CHECK: fir.call @_QPbar_derived
  call bar_derived(p)
end subroutine

! CHECK-LABEL: func @_QPderived_pointer_no_copy_array(
subroutine derived_pointer_no_copy_array(p)
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
