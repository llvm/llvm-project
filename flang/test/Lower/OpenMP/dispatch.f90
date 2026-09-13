!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HLFIR

! Variant selection is provided by DECLARE VARIANT with a `construct={dispatch}`
! match: inside a dispatch region the call to the base procedure `foo_dispatch`
! is replaced by a call to its variant `foo_variant`. `base_routine` additionally
! carries a `device={kind(host)}` variant to exercise re-resolution under
! `nocontext` when more than one variant matches.

module funcs
  implicit none

contains

  !HLFIR-LABEL: func @_QMfuncsPfoo_variant
  subroutine foo_variant()
    print *, "in foo_variant"
  end subroutine

  !HLFIR-LABEL: func @_QMfuncsPfoo_dispatch
  subroutine foo_dispatch()
    !$omp declare variant(foo_dispatch:foo_variant) match(construct={dispatch})
    print *, "in foo_dispatch"
  end subroutine

  !HLFIR-LABEL: func @_QMfuncsPdispatch_variant
  subroutine dispatch_variant()
    print *, "in dispatch_variant"
  end subroutine

  !HLFIR-LABEL: func @_QMfuncsPhost_variant
  subroutine host_variant()
    print *, "in host_variant"
  end subroutine

  ! `base_routine` has two variants: `dispatch_variant` matches
  ! `construct={dispatch}` and `host_variant` matches `device={kind(host)}`.
  !HLFIR-LABEL: func @_QMfuncsPbase_routine
  subroutine base_routine()
    !$omp declare variant(base_routine:dispatch_variant) match(construct={dispatch})
    !$omp declare variant(base_routine:host_variant) match(device={kind(host)})
    print *, "in base_routine"
  end subroutine

end module funcs

!HLFIR-LABEL: func @_QQmain
program dispatch_test
  use funcs
  implicit none
  logical :: cond

  ! A call outside any dispatch region targets the base procedure.
  !HLFIR: fir.call @_QMfuncsPfoo_dispatch() {{.*}}: () -> ()
  call foo_dispatch()

  !HLFIR: omp.dispatch {
  !$omp dispatch
  !HLFIR:   fir.call @_QMfuncsPfoo_variant() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }

  !HLFIR: omp.dispatch nowait {
  !$omp dispatch nowait
  !HLFIR:   fir.call @_QMfuncsPfoo_variant() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }

  ! novariants: runtime select of base/variant address, then indirect call, so
  ! the arguments are evaluated once.
  !HLFIR:   %[[COND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[COND_I1:.*]] = fir.convert %[[COND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch novariants(%[[COND_I1]]) {
  !$omp dispatch novariants(cond)
  !HLFIR:   %[[VARIANT:.*]] = fir.address_of(@_QMfuncsPfoo_variant) : () -> ()
  !HLFIR:   %[[BASE:.*]] = fir.address_of(@_QMfuncsPfoo_dispatch) : () -> ()
  !HLFIR:   %[[TARGET:.*]] = arith.select %[[COND_I1]], %[[BASE]], %[[VARIANT]] : () -> ()
  !HLFIR:   fir.call %[[TARGET]]() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }

  ! nocontext: the dispatch construct is dropped from the OpenMP context when
  ! the condition is true, so the same base/variant runtime select is emitted.
  !HLFIR:   %[[NCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[NCOND_I1:.*]] = fir.convert %[[NCOND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[NCOND_I1]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR:   %[[NVARIANT:.*]] = fir.address_of(@_QMfuncsPfoo_variant) : () -> ()
  !HLFIR:   %[[NBASE:.*]] = fir.address_of(@_QMfuncsPfoo_dispatch) : () -> ()
  !HLFIR:   %[[NTARGET:.*]] = arith.select %[[NCOND_I1]], %[[NBASE]], %[[NVARIANT]] : () -> ()
  !HLFIR:   fir.call %[[NTARGET]]() {{.*}}: () -> ()
    call foo_dispatch()
  !HLFIR:   omp.terminator
  !HLFIR: }

  ! nocontext with two matching variants: with the dispatch construct removed
  ! from the context, `construct={dispatch}` no longer matches and selection
  ! re-resolves to the `device={kind(host)}` variant, so the runtime select is
  ! between the two variants (not the base procedure).
  !HLFIR:   %[[MCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[MCOND_I1:.*]] = fir.convert %[[MCOND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[MCOND_I1]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR:   %[[MVARIANT:.*]] = fir.address_of(@_QMfuncsPdispatch_variant) : () -> ()
  !HLFIR:   %[[MHOST:.*]] = fir.address_of(@_QMfuncsPhost_variant) : () -> ()
  !HLFIR:   %[[MTARGET:.*]] = arith.select %[[MCOND_I1]], %[[MHOST]], %[[MVARIANT]] : () -> ()
  !HLFIR:   fir.call %[[MTARGET]]() {{.*}}: () -> ()
    call base_routine()
  !HLFIR:   omp.terminator
  !HLFIR: }
end program
