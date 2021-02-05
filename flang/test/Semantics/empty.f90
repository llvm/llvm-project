! RUN: %f18 -fsyntax-only %s
! RUN: rm -rf %t && mkdir %t
! RUN: touch %t/empty.f90
! RUN: %f18 -fsyntax-only %t/empty.f90
