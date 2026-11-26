! Check that "/*" doesn't trigger a C-style comment warning.
! RUN: %flang_fc1 -fsyntax-only -pedantic -Werror %s

    integer :: x
    x = 1
    write(*,10) x, x+1, x+2
10  format(1x,i2/*(i3))
end
