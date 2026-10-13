! RUN: %python %S/test_modfile.py %s %flang_fc1
! RUN: mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %s
! RUN: cd %t && printf '%%s\n' 'subroutine test' 'use m' 'integer, pointer :: local' 'type(t) :: object' 'k=1' 'object%%p=1' 'local=>getter()' 'call consume(k)' 'end' | not %flang_fc1 -fsyntax-only -I %t -x f95 - 2>&1 | FileCheck %s --check-prefix=READ
! Check that implicitly typed entities get a type in the module file.

module m
  public :: a
  private :: b
  protected :: i
  allocatable :: j
  pointer :: k, n
  protected_target k
  protected_target :: n
  type t
    integer, pointer, protected_target :: p
  end type
  interface
    subroutine consume(p)
      integer, pointer, protected_target, intent(in) :: p
    end subroutine
    function getter() result(p)
      integer, pointer, protected_target :: p
    end function
  end interface
end

!Expect: m.mod
!module m
! real(4)::a
! real(4),private::b
! integer(4),protected::i
! integer(4),allocatable::j
! integer(4),pointer,protected_target::k
! integer(4),pointer,protected_target::n
! type::t
!  integer(4),pointer,protected_target::p
! end type
! interface
!  subroutine consume(p)
!   integer(4),intent(in),pointer,protected_target::p
!  end
! end interface
! interface
!  function getter() result(p)
!   integer(4),pointer,protected_target::p
!  end
! end interface
!end

!READ: error: Semantic errors in -
!READ-NOT: error:
!READ: error: Left-hand side of assignment is not definable
!READ: because: 'k' has the PROTECTED_TARGET attribute and its target is not definable
!READ-NOT: error:
!READ: error: Left-hand side of assignment is not definable
!READ: because: 'p' has the PROTECTED_TARGET attribute and its target is not definable
!READ-NOT: error:
!READ: error: Target 'p' with PROTECTED_TARGET attribute may not be associated with pointer 'local' without PROTECTED_TARGET
!READ-NOT: error:
