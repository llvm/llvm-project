! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --allow-empty
! Regression test for pFUnit case: ensure that 0*ka doesn't get rewritten
! into a scalar 0 and then fail generic resolution.
! CHECK-NOT: error:
program test
  interface g
    procedure s
  end interface
  integer(1) a(1)
  a(1) = 2
  call test(1_1, a)
 contains
  subroutine s(a1,a2)
    integer(1) a1(:), a2(:)
    print *, a1
    print *, a2
  end
  subroutine test(j,ka)
    integer(1) j, ka(:)
    call g(int(j+0*ka,kind(ka)), ka)
  end
end
