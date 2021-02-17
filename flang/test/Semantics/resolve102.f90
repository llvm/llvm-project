! RUN: %S/test_errors.sh %s %t %f18

! Tests for circularly defined procedures
!ERROR: Procedure 'sub' is recursively defined.  Procedures in the cycle: ''sub', 'p2''
subroutine sub(p2)
  PROCEDURE(sub) :: p2

  call sub()
end subroutine

subroutine circular
  !ERROR: Procedure 'p' is recursively defined.  Procedures in the cycle: ''p', 'sub', 'p2''
  procedure(sub) :: p

  call p(sub)

  contains
    subroutine sub(p2)
      procedure(p) :: p2
    end subroutine
end subroutine circular

program iface
  !ERROR: Procedure 'p' is recursively defined.  Procedures in the cycle: ''p', 'sub', 'p2''
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
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: ''p', 'sub1', 'arg''
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
    !ERROR: Procedure 'sub1' is recursively defined.  Procedures in the cycle: ''p', 'sub1', 'arg', 'sub', 'p2''
    Subroutine sub1(arg)
      procedure(sub) :: arg
    End Subroutine

    Subroutine sub(p2)
      Procedure(sub1) :: p2
    End Subroutine
End Program
