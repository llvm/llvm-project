! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
subroutine test(dp1, dp2)
  intrinsic sin
  interface
    elemental real function elemental(x)
      real, intent(in) :: x
    end
    pure real function nonelemental(x)
      real, intent(in) :: x
    end
  end interface
  !PORTABILITY: A dummy procedure should not have an ELEMENTAL intrinsic as its interface
  procedure(sin) :: dp1
  !ERROR: A dummy procedure may not be ELEMENTAL
  procedure(elemental) :: dp2
  !PORTABILITY: Procedure pointer 'pp1' should not have an ELEMENTAL intrinsic as its interface
  procedure(sin), pointer :: pp1
  !ERROR: Procedure pointer 'pp2' may not be ELEMENTAL
  procedure(elemental), pointer :: pp2
  procedure(elemental) :: pp3 ! ok, external
  procedure(nonelemental), pointer :: pp4 => sin ! ok, special case
  !ERROR: Procedure pointer 'pp5' cannot be initialized with the elemental procedure 'elemental'
  procedure(nonelemental), pointer :: pp5 => elemental
end
