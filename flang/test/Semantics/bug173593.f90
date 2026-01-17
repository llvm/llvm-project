!RUN: %python %S/test_errors.py %s %flang_fc1
module m
 contains
  subroutine s1(x)
    integer :: x(..)
    !ERROR: An assumed-rank dummy argument may not be parenthesized
    call s2((x))
  end
  subroutine s2(y)
    integer :: y(..)
  end
end
