! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

module pure_not_enough_for_simple

  abstract interface
    simple subroutine simple_proc()
    end subroutine
  end interface

contains

  subroutine needs_simple(p)
    procedure(simple_proc) :: p
  end subroutine

  pure subroutine pure_only()
  end subroutine

  subroutine test()
    call needs_simple(pure_only)
  end subroutine

end module

! CHECK: incompatible procedure attributes
! CHECK: Simple
