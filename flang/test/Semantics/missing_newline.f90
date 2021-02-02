! RUN: echo -n "end program" > %t.f90
! RUN: %f18 -fsyntax-only %t.f90
! RUN: echo -ne "\rend program" > %t.f90
! RUN: %f18 -fsyntax-only %t.f90
