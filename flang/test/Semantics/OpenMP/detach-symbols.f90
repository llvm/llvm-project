! REQUIRES: openmp_runtime
! RUN: %python %S/../test_symbols.py %s %flang_fc1 %openmp_flags -fopenmp-version=50

! Test that variables in a DETACH clause get predetermined shared DSA.

!DEF: /detach_symbol_dsa (Subroutine) Subprogram
subroutine detach_symbol_dsa
  !DEF: /omp_lib (ModFile) Module
  !DEF: /omp_lib/omp_event_handle_kind PARAMETER, PUBLIC Use INTEGER(4)
  use :: omp_lib, only: omp_event_handle_kind
  !DEF: /detach_symbol_dsa/omp_event_handle_kind PARAMETER Use INTEGER(4)
  !DEF: /detach_symbol_dsa/ev (OmpShared, OmpPreDetermined) ObjectEntity INTEGER(8)
  integer(kind=omp_event_handle_kind) ev

  !$omp task detach(ev)
  !$omp end task
end subroutine detach_symbol_dsa
