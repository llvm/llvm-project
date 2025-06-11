module m
 contains

! Directive inside macro on same line; works
#define MACRO(X)  subroutine func1(X);    real(2) :: X;    !dir$ ignore_tkr(d) X; end subroutine func1;
MACRO(foo)

! Same subroutine, but after preprocessor expansion (-e -fno-reformat); syntax error
  ! subroutine func2(foo);  real(2) :: foo; !dir$ ignore_tkr(d) foo;  end subroutine func2;

! Parses with line wrap before !dir$
  subroutine func3(foo);     real(2) :: foo;
  !dir$ ignore_tkr(d) foo; end subroutine func3;

! Parses with line wrap after !dir$, but swallows the directive
  subroutine func4(foo); real(2) :: foo; !dir$ ignore_tkr(d) foo;
  end subroutine func4;

end module