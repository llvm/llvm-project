! RUN: %python %S/test_errors.py %s %flang_fc1

! Tests for circularly defined procedures
!ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: 'sub', 'p2'
subroutine sub(p2)
  PROCEDURE(sub) :: p2

  call sub()
end subroutine

subroutine circular
  !ERROR: Procedure 'p' is recursively defined.  Procedures in the cycle: 'p', 'sub', 'p2'
  procedure(sub) :: p

  call p(sub)

  contains
    subroutine sub(p2)
      procedure(p) :: p2
    end subroutine
end subroutine circular

program iface
  !ERROR: Procedure 'p' is recursively defined.  Procedures in the cycle: 'p', 'sub', 'p2'
  procedure(sub) :: p
  interface
    subroutine sub(p2)
      import p
      procedure(p) :: p2
    end subroutine
  end interface
  call p(sub)
end program

Program mutual
  Procedure(sub1) :: p

  Call p(sub)

  contains
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'p', 'sub1', 'arg'
    Subroutine sub1(arg)
      procedure(sub1) :: arg
    End Subroutine

    Subroutine sub(p2)
      Procedure(sub1) :: p2
    End Subroutine
End Program

Program mutual1
  Procedure(sub1) :: p

  Call p(sub)

  contains
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: 'p', 'sub1', 'arg', 'sub', 'p2'
    Subroutine sub1(arg)
      procedure(sub) :: arg
    End Subroutine

    Subroutine sub(p2)
      Procedure(sub1) :: p2
    End Subroutine
End Program

program twoCycle
  !ERROR: The interface for procedure 'p1' is recursively defined
  !ERROR: The interface for procedure 'p2' is recursively defined
  procedure(p1) p2
  procedure(p2) p1
  call p1
  call p2
end program

program threeCycle
  !ERROR: The interface for procedure 'p1' is recursively defined
  !ERROR: The interface for procedure 'p2' is recursively defined
  procedure(p1) p2
  !ERROR: The interface for procedure 'p3' is recursively defined
  procedure(p2) p3
  procedure(p3) p1
  call p1
  call p2
  call p3
end program

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
