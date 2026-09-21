! Test -f[no-]oacc-cuda-bind-c-cfi on a genuinely assumed-shape BIND(C) dummy.
! By default it is passed using a Fortran 2018 CFI descriptor. With the
! legacy nvfortran OpenACC/CUDA Fortran convention requested
! (-fno-oacc-cuda-bind-c-cfi), lowering warns that the argument will be
! passed by address instead, then fails: there is no way to recover the
! array's shape from a bare pointer, unlike explicit-shape and assumed-size
! dummies (see bindc-legacy-cfi-descriptor.f90), which are unaffected by
! this option because they never required a descriptor to begin with.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefix=DEFAULT
! RUN: not %flang_fc1 -fno-oacc-cuda-bind-c-cfi -emit-hlfir -o - %s 2>&1 | FileCheck %s --check-prefix=LEGACY

subroutine assumed_shape(a) bind(c)
  real, intent(in) :: a(:)
end subroutine
! DEFAULT-LABEL: func.func @assumed_shape(
! DEFAULT-SAME: %{{.*}}: !fir.box<!fir.array<?xf32>>

! LEGACY: warning:{{.*}}argument 'a' will be passed by address, not a Fortran 2018 CFI descriptor, because CFI descriptor support is disabled for this compilation
! LEGACY: error:{{.*}}cannot determine the shape of assumed-shape or deferred-shape dummy argument 'a': it was passed by address instead of a Fortran 2018 CFI descriptor (-fno-oacc-cuda-bind-c-cfi is only valid for dummy arguments whose shape is never queried)
