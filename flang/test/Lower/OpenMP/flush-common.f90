! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s | FileCheck %s

! Regression test to ensure that the name /c/ in the flush argument list is
! resolved to the common block symbol and common blocks are allowed in the
! flush argument list.

! CHECK: %[[GLBL:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.array<4xi8>>
  common /c/ x
  real :: x
! CHECK: omp.flush(%[[GLBL]] : !fir.ref<!fir.array<4xi8>>)
  !$omp flush(/c/)
end

