!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=HLFIR

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

!HLFIR-LABEL: func @_QPtest_dispatch_argument(
subroutine test_dispatch_argument(c1, c2)
  implicit none
  logical :: c1, c2
  integer :: result
  real :: real_result
  interface
    integer function argument_dispatch(value)
      integer, value :: value
    end function
    integer function argument_host(value)
      integer, value :: value
    end function
    integer function argument_base(value) result(output)
      import :: argument_dispatch, argument_host
      integer, value :: value
      !$omp declare variant(argument_base:argument_dispatch) match(construct={dispatch})
      !$omp declare variant(argument_base:argument_host) match(device={kind(host)})
    end function
    subroutine target_dispatch(value)
      integer, value :: value
    end subroutine
    subroutine target_host(value)
      integer, value :: value
    end subroutine
    subroutine target_base(value)
      import :: target_dispatch, target_host
      integer, value :: value
      !$omp declare variant(target_base:target_dispatch) match(construct={dispatch})
      !$omp declare variant(target_base:target_host) match(device={kind(host)})
    end subroutine
  end interface

  !HLFIR: omp.dispatch {
  !$omp dispatch
  !HLFIR: %[[ARG_RESULT:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.call @_QPtarget_dispatch(%[[ARG_RESULT]]) {{.*}}: (i32) -> ()
  !HLFIR-NEXT: omp.terminator
  call target_base(argument_base(3))

  !HLFIR: omp.dispatch nocontext(%[[ARG_C2:.*]]) novariants(%[[ARG_C1:.*]]) {
  !$omp dispatch novariants(c1) nocontext(c2)
  !HLFIR-NOT: arith.select
  !HLFIR: %[[BOTH_ARG_RESULT:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: %[[ARG_DISPATCH:.*]] = fir.address_of(@_QPtarget_dispatch) : (i32) -> ()
  !HLFIR-NEXT: %[[ARG_HOST:.*]] = fir.address_of(@_QPtarget_host) : (i32) -> ()
  !HLFIR-NEXT: %[[ARG_CONTEXT:.*]] = arith.select %[[ARG_C2]], %[[ARG_HOST]], %[[ARG_DISPATCH]] : (i32) -> ()
  !HLFIR-NEXT: %[[ARG_BASE:.*]] = fir.address_of(@_QPtarget_base) : (i32) -> ()
  !HLFIR-NEXT: %[[ARG_TARGET:.*]] = arith.select %[[ARG_C1]], %[[ARG_BASE]], %[[ARG_CONTEXT]] : (i32) -> ()
  !HLFIR-NEXT: fir.call %[[ARG_TARGET]](%[[BOTH_ARG_RESULT]]) {{.*}}: (i32) -> ()
  !HLFIR-NEXT: omp.terminator
  call target_base(argument_base(3))

  !HLFIR: omp.dispatch nocontext(%[[FUNC_C2:.*]]) novariants(%[[FUNC_C1:.*]]) {
  !$omp dispatch novariants(c1) nocontext(c2)
  !HLFIR-NOT: arith.select
  !HLFIR: %[[INNER_RESULT:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: %[[FUNC_DISPATCH:.*]] = fir.address_of(@_QPargument_dispatch) : (i32) -> i32
  !HLFIR-NEXT: %[[FUNC_HOST:.*]] = fir.address_of(@_QPargument_host) : (i32) -> i32
  !HLFIR-NEXT: %[[FUNC_CONTEXT:.*]] = arith.select %[[FUNC_C2]], %[[FUNC_HOST]], %[[FUNC_DISPATCH]] : (i32) -> i32
  !HLFIR-NEXT: %[[FUNC_BASE:.*]] = fir.address_of(@_QPargument_base) : (i32) -> i32
  !HLFIR-NEXT: %[[FUNC_TARGET:.*]] = arith.select %[[FUNC_C1]], %[[FUNC_BASE]], %[[FUNC_CONTEXT]] : (i32) -> i32
  !HLFIR-NEXT: %[[OUTER_RESULT:.*]] = fir.call %[[FUNC_TARGET]](%[[INNER_RESULT]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: hlfir.assign %[[OUTER_RESULT]] to %{{.*}} : i32, !fir.ref<i32>
  !HLFIR-NEXT: omp.terminator
  result = argument_base(argument_base(3))

  !HLFIR: omp.dispatch {
  !$omp dispatch
  !HLFIR: %[[CONVERT_INNER:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: %[[CONVERT_OUTER:.*]] = fir.call @_QPargument_dispatch(%[[CONVERT_INNER]]) {{.*}}: (i32) -> i32
  !HLFIR: fir.convert %[[CONVERT_OUTER]] : (i32) -> f32
  real_result = argument_base(argument_base(3))
end subroutine

! Check novariants and nocontext selection when an allocatable argument
! to an IGNORE_TKR(C) dummy requires a function-pointer cast.
!HLFIR-LABEL: func @_QPtest_dispatch_ignore_tkr(
!HLFIR-SAME: %[[CAST_C1_ARG:[^:]+]]: !fir.ref<!fir.logical<4>> {{.*}}, %[[CAST_C2_ARG:[^:]+]]: !fir.ref<!fir.logical<4>> {{.*}}, %[[VALUES_ARG:[^:]+]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
subroutine test_dispatch_ignore_tkr(c1, c2, values)
  implicit none
  logical :: c1, c2
  real, allocatable :: values(:)
  interface
    subroutine cast_dispatch(values)
      real, intent(in) :: values(:)
      !dir$ ignore_tkr(c) values
    end subroutine
    subroutine cast_host(values)
      real, intent(in) :: values(:)
      !dir$ ignore_tkr(c) values
    end subroutine
    subroutine cast_base(values)
      import :: cast_dispatch, cast_host
      real, intent(in) :: values(:)
      !dir$ ignore_tkr(c) values
      !$omp declare variant(cast_base:cast_dispatch) match(construct={dispatch})
      !$omp declare variant(cast_base:cast_host) match(device={kind(host)})
    end subroutine
  end interface

  !HLFIR: %[[CAST_C1:.*]]:2 = hlfir.declare %[[CAST_C1_ARG]]
  !HLFIR: %[[CAST_C2:.*]]:2 = hlfir.declare %[[CAST_C2_ARG]]
  !HLFIR: %[[VALUES:.*]]:2 = hlfir.declare %[[VALUES_ARG]]
  !HLFIR: omp.dispatch novariants(%[[CAST_NV:.*]]) {
  !$omp dispatch novariants(c1)
  !HLFIR: %[[NV_DISPATCH_ADDR:.*]] = fir.address_of(@_QPcast_dispatch) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NV_DISPATCH:.*]] = fir.convert %[[NV_DISPATCH_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[NV_BASE_ADDR:.*]] = fir.address_of(@_QPcast_base) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NV_BASE:.*]] = fir.convert %[[NV_BASE_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[NV_TARGET:.*]] = arith.select %[[CAST_NV]], %[[NV_BASE]], %[[NV_DISPATCH]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: fir.call %[[NV_TARGET]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: omp.terminator
  call cast_base(values)

  !HLFIR: omp.dispatch nocontext(%[[CAST_NC:.*]]) {
  !$omp dispatch nocontext(c2)
  !HLFIR: %[[NC_DISPATCH_ADDR:.*]] = fir.address_of(@_QPcast_dispatch) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NC_DISPATCH:.*]] = fir.convert %[[NC_DISPATCH_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[NC_HOST_ADDR:.*]] = fir.address_of(@_QPcast_host) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NC_HOST:.*]] = fir.convert %[[NC_HOST_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[NC_TARGET:.*]] = arith.select %[[CAST_NC]], %[[NC_HOST]], %[[NC_DISPATCH]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: fir.call %[[NC_TARGET]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: omp.terminator
  call cast_base(values)

  !HLFIR: %[[CAST_C2_LOAD:.*]] = fir.load %[[CAST_C2]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[CAST_C2_I1:.*]] = fir.convert %[[CAST_C2_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: %[[CAST_C1_LOAD:.*]] = fir.load %[[CAST_C1]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[CAST_C1_I1:.*]] = fir.convert %[[CAST_C1_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[CAST_C2_I1]]) novariants(%[[CAST_C1_I1]]) {
  !$omp dispatch novariants(c1) nocontext(c2)
  !HLFIR: %[[BOTH_DISPATCH_ADDR:.*]] = fir.address_of(@_QPcast_dispatch) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[CAST_DISPATCH:.*]] = fir.convert %[[BOTH_DISPATCH_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[BOTH_HOST_ADDR:.*]] = fir.address_of(@_QPcast_host) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[CAST_HOST:.*]] = fir.convert %[[BOTH_HOST_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[CAST_CONTEXT:.*]] = arith.select %[[CAST_C2_I1]], %[[CAST_HOST]], %[[CAST_DISPATCH]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: %[[BOTH_BASE_ADDR:.*]] = fir.address_of(@_QPcast_base) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[CAST_BASE:.*]] = fir.convert %[[BOTH_BASE_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: %[[CAST_TARGET:.*]] = arith.select %[[CAST_C1_I1]], %[[CAST_BASE]], %[[CAST_CONTEXT]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: fir.call %[[CAST_TARGET]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: omp.terminator
  call cast_base(values)
end subroutine

!HLFIR-DAG: func.func private @_QPexternal_variant()
!HLFIR-DAG: func.func private @_QPexternal_base()
!HLFIR-DAG: func.func private @_QPexternal_dispatch_func(i32) -> i32
!HLFIR-DAG: func.func private @_QPexternal_host_func(i32) -> i32
