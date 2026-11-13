! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Each consequent-arg of a conditional argument gets the diagnostic that it
! would get on its own: the portability warning when it needs no copy-out, and
! the temporary-copy warning when it does.
module m
 contains
  subroutine volatile_explicit(x)
    real, volatile :: x(5)
  end subroutine
  subroutine test_warning_per_consequent(flag)
    logical, intent(in) :: flag
    real :: a(10), b(10)
    !PORTABILITY: The array section 'a(2_8:6_8:1_8)' should not be associated with dummy argument 'x=' with VOLATILE attribute, unless the dummy is assumed-shape or assumed-rank [-Wportability]
    !WARNING: The array section 'b(::2_8)' should not be associated with dummy argument 'x=' with VOLATILE attribute, unless the dummy is assumed-shape or assumed-rank [-Wvolatile-or-asynchronous-temporary]
    call volatile_explicit((flag ? a(2:6) : b(::2)))
  end subroutine
end module
