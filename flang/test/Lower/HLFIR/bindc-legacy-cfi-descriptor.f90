! Test -f[no-]oacc-cuda-bind-c-cfi: whether an assumed-shape, deferred-shape,
! or assumed-rank BIND(C) dummy argument is passed using a Fortran 2018 CFI
! descriptor (the default) or, for backward compatibility with the legacy
! nvfortran OpenACC/CUDA Fortran convention, by bare address instead.
!
! Explicit-shape and assumed-size dummies never required a descriptor in the
! first place (F2018 18.3.6), and allocatable/pointer dummies always require
! one regardless of shape, so all three are unaffected either way: same
! checks apply to both RUN lines below, and none should produce a warning.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -fno-oacc-cuda-bind-c-cfi -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine explicit_shape(a, n) bind(c)
  integer, intent(in), value :: n
  real, intent(in) :: a(n)
end subroutine
! CHECK-LABEL: func.func @explicit_shape(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<?xf32>>
! CHECK-NOT: will be passed by address

subroutine assumed_size(a) bind(c)
  real, intent(in) :: a(*)
end subroutine
! CHECK-LABEL: func.func @assumed_size(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<?xf32>>
! CHECK-NOT: will be passed by address

subroutine allocatable_dummy(a) bind(c)
  real, intent(in), allocatable :: a(:)
end subroutine
! CHECK-LABEL: func.func @allocatable_dummy(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK-NOT: will be passed by address
