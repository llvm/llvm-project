! RUN: %python %S/test_errors.py %s %flang_fc1

! Tests for circularly defined procedures
!ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: 'sub', 'p2'
subroutine sub(p2)
  PROCEDURE(sub) :: p2
end subroutine

subroutine circular
  procedure(sub) :: p
  contains
    !ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: 'p', 'sub', 'p2'
    subroutine sub(p2)
      procedure(p) :: p2
    end subroutine
end subroutine circular

!ERROR: Procedure 'foo' is recursively defined.  Procedures in the cycle: 'foo', 'r'
function foo() result(r)
  !ERROR: Procedure 'r' is recursively defined.  Procedures in the cycle: 'foo', 'r'
  procedure(foo), pointer :: r 
end function foo

subroutine iface
  !ERROR: Procedure 'p' is recursively defined.  Procedures in the cycle: 'p', 'sub', 'p2'
  procedure(sub) :: p
  interface
    !ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: 'p', 'sub', 'p2'
    subroutine sub(p2)
      import p
      procedure(p) :: p2
    end subroutine
  end interface
  call p(sub)
end subroutine

subroutine mutual
  Procedure(sub1) :: p
  contains
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'p', 'sub1', 'arg'
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'sub1', 'arg', 'sub', 'p2'
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'sub1', 'arg'
    Subroutine sub1(arg)
      procedure(sub1) :: arg
    End Subroutine

    Subroutine sub(p2)
      Procedure(sub1) :: p2
    End Subroutine
End subroutine

subroutine mutual1
  Procedure(sub1) :: p
  contains
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'p', 'sub1', 'arg', 'sub', 'p2'
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'sub1', 'arg', 'sub', 'p2'
    Subroutine sub1(arg)
      procedure(sub) :: arg
    End Subroutine

    !ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: 'sub1', 'arg', 'sub', 'p2'
    Subroutine sub(p2)
      Procedure(sub1) :: p2
    End Subroutine
End subroutine

subroutine twoCycle
  !ERROR: The interface for procedure 'p1' is recursively defined
  !ERROR: The interface for procedure 'p2' is recursively defined
  procedure(p1) p2
  procedure(p2) p1
end subroutine

subroutine threeCycle
  !ERROR: The interface for procedure 'p1' is recursively defined
  !ERROR: The interface for procedure 'p2' is recursively defined
  procedure(p1) p2
  !ERROR: The interface for procedure 'p3' is recursively defined
  procedure(p2) p3
  procedure(p3) p1
end subroutine

module mutualSpecExprs
contains
  pure integer function f(n)
    integer, intent(in) :: n
    real arr(g(n))
    f = size(arr)
  end function
  pure integer function g(n)
    integer, intent(in) :: n
    !ERROR: Procedure 'f' is referenced before being sufficiently defined in a context where it must be so
    real arr(f(n))
    g = size(arr)
  end function
end

module genericInSpec
  interface int
    procedure ifunc
  end interface
 contains
  function ifunc(x)
    integer a(int(kind(1))) ! generic is ok with most compilers
    integer(size(a)), intent(in) :: x
    ifunc = x
  end
end
