! RUN: echo -n "end program" > %t.f90
! RUN: %flang_fc1 -fsyntax-only %t.f90
! RUN: echo -n -e "\rend program" > %t.f90
! RUN: %flang_fc1 -fsyntax-only %t.f90
