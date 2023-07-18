! RUN: %python %S/test_errors.py %s %flang_fc1
module m
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
    rf(x) = 3.
  end
  subroutine test3
    external sf
    !ERROR: 'sf' has not been declared as an array or pointer-valued function
    sf(x) = 4.
  end
end
