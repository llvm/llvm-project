! RUN: split-file %s %t
! RUN: %flang_fc1 -fdebug-unparse %t/integer.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse -funsigned %t/unsigned.f90 2>&1 | FileCheck %s --check-prefix=UNSIGNED
! max(c, max(c, x)) folds to max(c, x), and likewise for min, in any operand
! order, when the repeated operand c is a constant.  Besides being a valid
! simplification, this makes an extent written to a module file as
! max(0,ubound(a,dim)) read back as the same expression tree: re-analyzing
! ubound() re-expands to max(0,...) and would otherwise wrap one more level on
! every round trip.
!
! Only a constant is absorbed.  Expr::operator== is defined for comparing
! procedure characteristics and treats two dummy arguments at the same position
! as equal, so a repeated variable is not reliably the same variable; and a
! repeated impure function reference or VOLATILE read must be evaluated twice.
!--- integer.f90
subroutine s(n, m)
  integer, intent(in) :: n, m
  ! All four orientations of a repeated constant.
  !CHECK: PRINT *, max(0_4,n)
  print *, max(0, max(0, n))
  !CHECK: PRINT *, max(n,0_4)
  print *, max(0, max(n, 0))
  !CHECK: PRINT *, max(0_4,n)
  print *, max(max(0, n), 0)
  !CHECK: PRINT *, max(n,0_4)
  print *, max(max(n, 0), 0)
  !CHECK: PRINT *, max(0_4,n)
  print *, max(0, max(0, max(0, n)))
  !CHECK: PRINT *, min(7_4,n)
  print *, min(7, min(7, n))
  ! A repeated non-constant operand does not collapse.
  !CHECK: PRINT *, max(m,max(m,n))
  print *, max(m, max(m, n))
  !CHECK: PRINT *, max(max(n,m),m)
  print *, max(max(n, m), m)
  !CHECK: PRINT *, min(n+1_4,min(n+1_4,m))
  print *, min(n + 1, min(n + 1, m))
  ! Different constants do not collapse.
  !CHECK: PRINT *, max(0_4,max(1_4,n))
  print *, max(0, max(1, n))
  ! Mixed min/max does not collapse.
  !CHECK: PRINT *, max(0_4,min(0_4,n))
  print *, max(0, min(0, n))
  !CHECK: PRINT *, min(m,max(m,n))
  print *, min(m, max(m, n))
end subroutine

! A host procedure's dummy and an internal procedure's dummy both occupy
! argument position 1 and so compare equal under Expr::operator==; absorbing
! one would print b instead of max(a,b) when a > b.
subroutine outer(a)
  integer, intent(in) :: a
  call inner(1)
contains
  subroutine inner(b)
    integer, intent(in) :: b
    !CHECK: PRINT *, max(a,max(b,0_4))
    print *, max(a, max(b, 0))
    !CHECK: PRINT *, max(max(b,0_4),a)
    print *, max(max(b, 0), a)
    !CHECK: PRINT *, min(a,min(b,9_4))
    print *, min(a, min(b, 9))
  end subroutine
end subroutine

! Both references to an impure function, and both reads of a VOLATILE
! variable, are kept.
subroutine effects(v)
  integer, volatile :: v
  integer :: r
  !CHECK: r=max(tick(),max(tick(),0_4))
  r = max(tick(), max(tick(), 0))
  !CHECK: PRINT *, max(v,max(v,0_4))
  print *, max(v, max(v, 0))
contains
  integer function tick()
    integer, save :: calls = 0
    calls = calls + 1
    tick = calls
  end function
end subroutine

!--- unsigned.f90
! The UNSIGNED arm of the fold: a repeated unsigned constant is absorbed in
! either operand order; a repeated unsigned variable is not.
subroutine s(n, m)
  unsigned, intent(in) :: n, m
  !UNSIGNED: PRINT *, max(5U_4,n)
  print *, max(5u, max(5u, n))
  !UNSIGNED: PRINT *, max(n,5U_4)
  print *, max(max(n, 5u), 5u)
  !UNSIGNED: PRINT *, min(5U_4,n)
  print *, min(5u, min(5u, n))
  !UNSIGNED: PRINT *, max(m,max(m,n))
  print *, max(m, max(m, n))
  !UNSIGNED: PRINT *, max(max(m,n),m)
  print *, max(max(m, n), m)
end subroutine
