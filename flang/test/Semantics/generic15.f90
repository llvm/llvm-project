! RUN: %flang_fc1 -fsyntax-only %s
! Regression test for #191407

module m
  interface gen
    module procedure with_sub, with_fun
  end interface gen
contains
  subroutine with_sub(sub, n)
    external sub
    call sub(n)
  end subroutine with_sub
  subroutine with_fun(af, n)
    interface
      function af(n)
        real af(n)
      end function
    end interface
    print *,'[',af(n),']'
  end subroutine
end module

program test
  use m
  external s
  integer a

  a = 13
  call gen(s, a)
end program

subroutine s(n)
  if (n == 13) print '("ok")'
end subroutine
