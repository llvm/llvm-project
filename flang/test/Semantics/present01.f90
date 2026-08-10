! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type dt
    real a
  end type
 contains
  subroutine s(a,b,p,unl)
    type(dt), optional :: a(:), b
    procedure(sin), optional :: p
    type(*), optional :: unl
    print *, present(a) ! ok
    print *, present(p) ! ok
    print *, present(unl) ! ok
    !ERROR: Argument of PRESENT() must be the name of a whole OPTIONAL dummy argument
    print *, present(a(1))
    !ERROR: Argument of PRESENT() must be the name of a whole OPTIONAL dummy argument
    print *, present(b%a)
    !ERROR: Argument of PRESENT() must be the name of a whole OPTIONAL dummy argument
    print *, present(a(1)%a)
  end
end
