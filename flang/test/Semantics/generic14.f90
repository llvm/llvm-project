! RUN: %flang_fc1 -fsyntax-only %s
! Regression test for #191404)

module m
  interface gen
    subroutine with_real(freal)
      real, external :: freal
    end subroutine with_real
    subroutine with_complex(fcom)
      complex, external :: fcom
    end subroutine with_complex
  end interface
end module
subroutine with_real(f)
  real, external :: f
  print '("ok",F12.7)', f()
end subroutine
subroutine with_complex(f)
  complex, external :: f
  print '("fail",F12.7)', f()
end subroutine
program test
  use m
  external fact
  call gen(fact)
  print '("ok")'
end program

function fact()
  real :: fact
  fact = 42.0
end function fact
