! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1521
! If proc-language-binding-spec (bind(c)) is specified, the proc-interface
! shall appear, it shall be an interface-name, and interface-name shall be
! declared with a proc-language-binding-spec.

module m

  interface
    subroutine proc1() bind(c)
    end
    subroutine proc2()
    end
  end interface

  interface proc3
    subroutine proc3() bind(c)
    end
  end interface

  procedure(proc1), bind(c) :: pc1 ! no error
  procedure(proc3), bind(c) :: pc4 ! no error

  !ERROR: An interface name with BIND attribute must be specified if the BIND attribute is specified in a procedure declaration statement
  procedure(proc2), bind(c) :: pc2

  !ERROR: An interface name with BIND attribute must be specified if the BIND attribute is specified in a procedure declaration statement
  procedure(integer), bind(c) :: pc3

end
