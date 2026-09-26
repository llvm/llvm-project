!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HLFIR

! Variant selection is provided by DECLARE VARIANT with a `construct={dispatch}`
! match: inside a dispatch region the call to the base procedure `foo_dispatch`
! is replaced by a call to its variant `foo_variant`. `base_routine` additionally
! carries a `device={kind(host)}` variant to exercise re-resolution under
! `nocontext` when more than one variant matches.

module funcs
  implicit none

contains

  !novariants clause : base & variant subroutines
  !HLFIR-LABEL: func @_QMfuncsPfoo_variant
  subroutine foo_variant()
    print *, "in foo_variant"
  end subroutine

  !HLFIR-LABEL: func @_QMfuncsPfoo_dispatch
  subroutine foo_dispatch()
    !$omp declare variant(foo_dispatch:foo_variant) match(construct={dispatch})
    print *, "in foo_dispatch"
  end subroutine

  !nocontext clause : base & variant subroutines
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

!HLFIR-LABEL: func @_QPtest_novariants_nocontext(
!HLFIR-SAME: %[[C1_ARG:[^:]+]]: !fir.ref<!fir.logical<4>> {{.*}}, %[[C2_ARG:[^:]+]]: !fir.ref<!fir.logical<4>>
subroutine test_novariants_nocontext(c1, c2)
  use funcs
  implicit none
  logical :: c1, c2

  !HLFIR: %[[C1:.*]]:2 = hlfir.declare %[[C1_ARG]]
  !HLFIR: %[[C2:.*]]:2 = hlfir.declare %[[C2_ARG]]
  !HLFIR: %[[C2_LOAD:.*]] = fir.load %[[C2]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[C2_I1:.*]] = fir.convert %[[C2_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: %[[C1_LOAD:.*]] = fir.load %[[C1]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[C1_I1:.*]] = fir.convert %[[C1_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[C2_I1]]) novariants(%[[C1_I1]]) {
  !$omp dispatch novariants(c1) nocontext(c2)
  !HLFIR: %[[BOTH_VARIANT:.*]] = fir.address_of(@_QMfuncsPdispatch_variant) : () -> ()
  !HLFIR: %[[BOTH_HOST:.*]] = fir.address_of(@_QMfuncsPhost_variant) : () -> ()
  !HLFIR: %[[CONTEXT_TARGET:.*]] = arith.select %[[C2_I1]], %[[BOTH_HOST]], %[[BOTH_VARIANT]] : () -> ()
  !HLFIR: %[[BOTH_BASE:.*]] = fir.address_of(@_QMfuncsPbase_routine) : () -> ()
  !HLFIR: %[[BOTH_TARGET:.*]] = arith.select %[[C1_I1]], %[[BOTH_BASE]], %[[CONTEXT_TARGET]] : () -> ()
  !HLFIR: fir.call %[[BOTH_TARGET]]() {{.*}}: () -> ()
  call base_routine()
  !HLFIR: omp.terminator
  !HLFIR: }
end subroutine

!HLFIR-LABEL: func @_QPtest_external_novariants(
subroutine test_external_novariants(cond)
  implicit none
  logical :: cond
  interface
    subroutine external_variant()
    end subroutine
    subroutine external_base()
      import :: external_variant
      !$omp declare variant(external_base:external_variant) match(construct={dispatch})
    end subroutine
  end interface

  !HLFIR: omp.dispatch novariants(%[[EXT_COND:.*]]) {
  !$omp dispatch novariants(cond)
  !HLFIR: %[[EXT_VARIANT:.*]] = fir.address_of(@_QPexternal_variant) : () -> ()
  !HLFIR: %[[EXT_BASE:.*]] = fir.address_of(@_QPexternal_base) : () -> ()
  !HLFIR: %[[EXT_TARGET:.*]] = arith.select %[[EXT_COND]], %[[EXT_BASE]], %[[EXT_VARIANT]] : () -> ()
  !HLFIR-NEXT: fir.call %[[EXT_TARGET]]() {{.*}}: () -> ()
  call external_base()
  !HLFIR-NEXT: omp.terminator
end subroutine

!HLFIR-LABEL: func @_QPtest_external_nocontext(
!HLFIR-SAME: %[[EXT_COND_ARG:[^:]+]]: !fir.ref<!fir.logical<4>> {{.*}}, %[[EXT_VALUE_ARG:[^:]+]]: !fir.ref<i32>
integer function test_external_nocontext(cond, value) result(res)
  implicit none
  logical :: cond
  integer :: value
  interface
    integer function external_dispatch_func(value)
      integer, value :: value
    end function
    integer function external_host_func(value)
      integer, value :: value
    end function
    integer function external_base_func(value) result(output)
      import :: external_dispatch_func, external_host_func
      integer, value :: value
      !$omp declare variant(external_base_func:external_dispatch_func) match(construct={dispatch})
      !$omp declare variant(external_base_func:external_host_func) match(device={kind(host)})
    end function
  end interface

  !HLFIR: %[[EXT_COND_ADDR:.*]]:2 = hlfir.declare %[[EXT_COND_ARG]]
  !HLFIR: %[[EXT_RESULT_ADDR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_external_nocontextEres"}
  !HLFIR: %[[EXT_VALUE_ADDR:.*]]:2 = hlfir.declare %[[EXT_VALUE_ARG]]
  !HLFIR: %[[EXT_COND_LOAD:.*]] = fir.load %[[EXT_COND_ADDR]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[EXT_NCOND:.*]] = fir.convert %[[EXT_COND_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[EXT_NCOND]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR: %[[EXT_VALUE:.*]] = fir.load %[[EXT_VALUE_ADDR]]#0 : !fir.ref<i32>
  !HLFIR: %[[EXT_DISPATCH:.*]] = fir.address_of(@_QPexternal_dispatch_func) : (i32) -> i32
  !HLFIR: %[[EXT_HOST:.*]] = fir.address_of(@_QPexternal_host_func) : (i32) -> i32
  !HLFIR: %[[EXT_NTARGET:.*]] = arith.select %[[EXT_NCOND]], %[[EXT_HOST]], %[[EXT_DISPATCH]] : (i32) -> i32
  !HLFIR-NEXT: %[[EXT_RESULT:.*]] = fir.call %[[EXT_NTARGET]](%[[EXT_VALUE]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: hlfir.assign %[[EXT_RESULT]] to %[[EXT_RESULT_ADDR]]#0 : i32, !fir.ref<i32>
  res = external_base_func(value)
  !HLFIR-NEXT: omp.terminator
  !HLFIR: %[[EXT_RETURN:.*]] = fir.load %[[EXT_RESULT_ADDR]]#0 : !fir.ref<i32>
  !HLFIR: return %[[EXT_RETURN]] : i32
end function

!HLFIR-DAG: func.func private @_QPexternal_variant()
!HLFIR-DAG: func.func private @_QPexternal_base()
!HLFIR-DAG: func.func private @_QPexternal_dispatch_func(i32) -> i32
!HLFIR-DAG: func.func private @_QPexternal_host_func(i32) -> i32
