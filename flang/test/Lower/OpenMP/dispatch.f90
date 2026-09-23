!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=HLFIR
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s --check-prefix=HLFIR
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s --check-prefix=HLFIR

! Dispatch lowers to a no-op omp.dispatch region wrapping the associated call.
! Variant selection and the nocontext/novariants clauses are added separately;
! here the call inside the region targets the base procedure unchanged.

module funcs
  implicit none

contains

  !HLFIR-LABEL: func @_QMfuncsPfoo_dispatch
  subroutine foo_dispatch()
    print *, "in foo_dispatch"
  end subroutine

end module funcs

!HLFIR-LABEL: func @_QQmain
program dispatch_test
  use funcs
  implicit none

  !HLFIR: omp.dispatch {
  !$omp dispatch
  !HLFIR:   fir.call @_QMfuncsPfoo_dispatch() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }

  !HLFIR: omp.dispatch nowait {
  !$omp dispatch nowait
  !HLFIR:   fir.call @_QMfuncsPfoo_dispatch() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }
end program
