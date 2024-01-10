! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! A nasty case of a weird order of declarations - a symbol may appear
! as an actual argument to a specification function before its rank
! has been declared.
program main
  interface kind
    pure integer function mykind(x)
      real, intent(in) :: x(:)
    end
  end interface
  real a, b
  integer, parameter :: ak = kind(a)
  integer, parameter :: br = rank(b)
  !WARNING: 'a' appeared earlier as a scalar actual argument to a specification function
  dimension a(1)
  !WARNING: 'b' appeared earlier as a scalar actual argument to a specification function
  dimension b(1)
end
