! RUN: %python %S/test_errors.py %s %flang_fc1
! Enforce array conformance across actual arguments to ELEMENTAL
module m
 contains
  real elemental function f(a, b)
    real, intent(in) :: a, b
    f = a + b
  end function
  real function g(n)
    integer, value :: n
    g = sqrt(real(n))
  end function
  subroutine test
    real :: a(3) = [1, 2, 3]
    !ERROR: Dimension 1 of actual argument (a) corresponding to dummy argument #1 ('a') has extent 3, but actual argument ([REAL(4)::(g(int(j,kind=4)),INTEGER(8)::j=1_8,2_8,1_8)]) corresponding to dummy argument #2 ('b') has extent 2
    print *, f(a, [(g(j), j=1, 2)])
  end subroutine
end
