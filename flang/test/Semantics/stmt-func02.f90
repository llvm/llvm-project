! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m1
 contains
  real function rf2(x)
    rf2 = x
  end
end
module m2
  use m1
  real, target :: x = 1.
 contains
  function rpf(x)
    real, intent(in out), target :: x
    real, pointer :: rpf
    rpf => x
  end
  real function rf(x)
    rf = x
  end
  subroutine test1
    ! This is a valid assignment, not a statement function.
    ! Every other Fortran compiler misinterprets it!
    rpf(x) = 2. ! statement function or indirect assignment?
    print *, x
  end
  subroutine test2
    !PORTABILITY: Name 'rf' from host scope should have a type declaration before its local statement function definition
    rf(x) = 1.
  end
  subroutine test2b
    !PORTABILITY: Name 'rf2' from host scope should have a type declaration before its local statement function definition
    rf2(x) = 1.
  end
  subroutine test3
    external sf
    !ERROR: 'sf' has not been declared as an array or pointer-valued function
    sf(x) = 4.
  end
  function f()
    !ERROR: Recursive call to 'f' requires a distinct RESULT in its declaration
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'f()' is not a variable or pointer
    f() = 1. ! statement function of same name as function
  end
  function g() result(r)
    !WARNING: Name 'g' from host scope should have a type declaration before its local statement function definition
    !ERROR: 'g' is already declared in this scoping unit
    g() = 1. ! statement function of same name as function
  end
  function h1() result(r)
    !ERROR: 'r' is not a callable procedure
    r() = 1. ! statement function of same name as function result
  end
  function h2() result(r)
    procedure(real), pointer :: r
    r() = 1. ! not a statement function
  end
end
