! RUN: split-file %s %t
! RUN: %python %S/test_modfile.py %t/m.f90 %flang_fc1
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %python %S/test_errors.py %t/use.f90 %flang_fc1 -I%t -pedantic

!--- m.f90
module protected_target_mod
  integer, pointer, protected_target :: p
  integer, pointer :: q
  protected_target q
  type :: t
    integer, pointer, protected_target :: c
  end type
contains
  subroutine accept(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => p
  end function
end module

!Expect: protected_target_mod.mod
!module protected_target_mod
!integer(4),pointer,protected_target::p
!integer(4),pointer,protected_target::q
!type::t
!integer(4),pointer,protected_target::c
!end type
!contains
!subroutine accept(a)
!integer(4),intent(in),pointer,protected_target::a
!end
!function get() result(r)
!integer(4),pointer,protected_target::r
!end
!end

!--- use.f90
subroutine use_protected_target()
  use protected_target_mod, only: local => p, q, t, get, accept
  interface
    subroutine pointer_in(p)
      integer, pointer, intent(in) :: p
    end subroutine
    subroutine value_inout(p)
      integer, intent(inout) :: p
    end subroutine
  end interface
  external :: implicit_interface
  integer, pointer :: writable
  integer, pointer, protected_target :: readonly
  type(t) :: object
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'local' is not definable
  local = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'q' is not definable
  q = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'c' is not definable
  object%c = 1
  !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'local'
  writable => local
  !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'r'
  writable => get()
  readonly => local
  readonly => get()
  call accept(local)
  call accept(writable)
  call accept(get())
  !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'local' or its subobject
  call pointer_in(local)
  !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'local' or its subobject
  call value_inout(local)
  !ERROR: PROTECTED_TARGET pointer 'local' or its subobject may not be passed to a procedure with an implicit interface
  call implicit_interface(local)
end subroutine
