! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/def.f90
! RUN: not %flang_fc1 -fsyntax-only -module-dir %t %t/use.f90 2>&1 | FileCheck %s

!--- def.f90
module mod_pt_def
  type sub_t
    integer :: a
    integer, pointer, protected_target :: pt_comp
  end type
  integer, pointer, protected_target :: p_var
contains
  subroutine sub_pt_dummy(arg)
    integer, pointer, protected_target :: arg
  end subroutine
  subroutine sub_ord_dummy(arg)
    integer, pointer :: arg
  end subroutine
  subroutine sub_no_intent(arg)
    integer :: arg
  end subroutine
end module

!--- use.f90
subroutine test_use_readback()
  use mod_pt_def
  integer, target :: tgt = 1
  integer, pointer :: ord_p
  type(sub_t) :: s

  ! C866: assignment through read-back PROTECTED_TARGET pointer variable
  ! CHECK: error: Left-hand side of assignment is not definable
  ! CHECK: because: The target of 'p_var' with the PROTECTED_TARGET attribute may not be defined
  p_var = 10

  ! C866: assignment through read-back PROTECTED_TARGET component
  ! CHECK: error: Left-hand side of assignment is not definable
  ! CHECK: because: The target of 'pt_comp' with the PROTECTED_TARGET attribute may not be defined
  s%pt_comp = 20

  ! C870: ordinary pointer target dropped protection
  ! CHECK: error: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
  ord_p => p_var

  ! C870: component form
  ! CHECK: error: Data pointer object must have the PROTECTED_TARGET attribute when the target is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
  ord_p => s%pt_comp

  ! C872: passing PROTECTED_TARGET to non-PROTECTED_TARGET pointer dummy
  ! CHECK: error: dummy argument 'arg=' must have the PROTECTED_TARGET attribute when the actual argument is a pointer with the PROTECTED_TARGET attribute or a subobject thereof
  call sub_ord_dummy(p_var)

  ! C872: passing PROTECTED_TARGET actual to PROTECTED_TARGET dummy is conforming
  call sub_pt_dummy(p_var)

  ! C874: passing PROTECTED_TARGET to non-pointer dummy without INTENT(IN)
  ! CHECK: error: Actual argument associated with dummy argument 'arg=' is a pointer with the PROTECTED_TARGET attribute (or a subobject thereof) and must be associated with an INTENT(IN) dummy argument
  call sub_no_intent(p_var)
end subroutine
