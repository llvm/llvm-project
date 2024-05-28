!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

!PORTABILITY: An interoperable procedure should have an interface
subroutine subr1(e) bind(c)
  external e
end

subroutine subr2(p) bind(c)
  !PORTABILITY: An interoperable procedure should have an interface
  procedure() :: p
end

subroutine subr3(p) bind(c)
  !PORTABILITY: An interoperable procedure should have an interface
  procedure(real) :: p
end

subroutine subr4(p) bind(c)
  interface
    !PORTABILITY: A dummy procedure of an interoperable procedure should be BIND(C)
    subroutine p(n)
      integer, intent(in) :: n
    end
  end interface
end

subroutine subr5(p) bind(c)
  interface
    !WARNING: A dummy procedure of an interoperable procedure should be BIND(C)
    subroutine p(c)
      character(*), intent(in) :: c
    end
  end interface
end

subroutine subr6(p) bind(c)
  interface
    function p()
      !ERROR: Interoperable function result must be scalar
      real p(1)
    end
  end interface
end

subroutine subr7(p) bind(c)
  interface
    !ERROR: Interoperable character function result must have length one
    character(*) function p()
    end
  end interface
end

subroutine subr8(p) bind(c)
  interface
    !WARNING: A dummy procedure of an interoperable procedure should be BIND(C)
    subroutine p(n)
      integer, intent(in), value :: n
    end
  end interface
end

subroutine subr9(p) bind(c)
  !ERROR: An interface name with the BIND attribute must appear if the BIND attribute appears in a procedure declaration
  procedure(q), bind(c), pointer :: p
  interface
    function q()
      real q(1)
    end
  end interface
end
