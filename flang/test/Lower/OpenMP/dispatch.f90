!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix=HLFIR

! Variant selection is provided by DECLARE VARIANT with a `construct={dispatch}`
! match: inside a dispatch region the call to the base procedure `foo_dispatch`
! is replaced by a call to its variant `foo_variant`. `base_routine` additionally
! carries a `device={kind(host)}` variant to exercise re-resolution under
! `nocontext` when more than one variant matches.

module funcs
  implicit none

contains

  !HLFIR-LABEL: func @_QMfuncsPfoo_no_variant
  subroutine foo_no_variant()
  end subroutine

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
    !$omp declare variant(base_routine:host_variant) match(device={kind(host)})
    !$omp declare variant(base_routine:dispatch_variant) match(construct={dispatch}, user={condition(score(2): .true.)})
    print *, "in base_routine"
  end subroutine

end module funcs

!HLFIR-LABEL: func @_QQmain
program dispatch_test
  use funcs
  implicit none
  logical :: cond

  !HLFIR: omp.dispatch {
  !$omp dispatch
  !HLFIR: fir.call @_QMfuncsPfoo_no_variant() {{.*}}: () -> ()
  call foo_no_variant()
  !HLFIR-NEXT: omp.terminator
  !HLFIR: }

  !HLFIR: omp.dispatch nowait {
  !$omp dispatch nowait
  !HLFIR: fir.call @_QMfuncsPfoo_no_variant() {{.*}}: () -> ()
  call foo_no_variant()
  !HLFIR-NEXT: omp.terminator
  !HLFIR: }

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

  ! novariants: a runtime branch between direct calls to the base and the
  ! variant; the arguments are evaluated once, before the branch.
  !HLFIR:   %[[COND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[COND_I1:.*]] = fir.convert %[[COND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch novariants(%[[COND_I1]]) {
  !$omp dispatch novariants(cond)
  !HLFIR:   fir.if %[[COND_I1]] {
  !HLFIR-NEXT: fir.call @_QMfuncsPfoo_dispatch() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QMfuncsPfoo_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: }
    call foo_dispatch()
  !HLFIR-NEXT: omp.terminator
  !HLFIR: }

  ! nocontext: the dispatch construct is dropped from the OpenMP context when
  ! the condition is true, so the same base/variant branch is emitted.
  !HLFIR:   %[[NCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[NCOND_I1:.*]] = fir.convert %[[NCOND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[NCOND_I1]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR:   fir.if %[[NCOND_I1]] {
  !HLFIR-NEXT: fir.call @_QMfuncsPfoo_dispatch() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QMfuncsPfoo_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: }
    call foo_dispatch()
  !HLFIR-NEXT: omp.terminator
  !HLFIR: }

  ! nocontext with two matching variants: with the dispatch construct removed
  ! from the context, `construct={dispatch}` no longer matches and selection
  ! re-resolves to the `device={kind(host)}` variant, so the runtime branch is
  ! between the two variants (not the base procedure).
  !HLFIR:   %[[MCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
  !HLFIR:   %[[MCOND_I1:.*]] = fir.convert %[[MCOND]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[MCOND_I1]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR:   fir.if %[[MCOND_I1]] {
  !HLFIR-NEXT: fir.call @_QMfuncsPhost_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QMfuncsPdispatch_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: }
    call base_routine()
  !HLFIR-NEXT: omp.terminator
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
  !HLFIR: fir.if %[[C1_I1]] {
  !HLFIR-NEXT: fir.call @_QMfuncsPbase_routine() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.if %[[C2_I1]] {
  !HLFIR-NEXT: fir.call @_QMfuncsPhost_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QMfuncsPdispatch_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: }
  !HLFIR-NEXT: }
  call base_routine()
  !HLFIR-NEXT: omp.terminator
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
  !HLFIR-NEXT: fir.if %[[EXT_COND]] {
  !HLFIR-NEXT: fir.call @_QPexternal_base() {{.*}}: () -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QPexternal_variant() {{.*}}: () -> ()
  !HLFIR-NEXT: }
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
      !$omp declare variant(external_base_func:external_dispatch_func) match(construct={dispatch}, user={condition(score(2): .true.)})
      !$omp declare variant(external_base_func:external_host_func) match(device={kind(host)})
    end function
  end interface

  !HLFIR: %[[EXT_COND_ADDR:.*]]:2 = hlfir.declare %[[EXT_COND_ARG]]
  !HLFIR: %[[EXT_RESULT_ADDR:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_external_nocontextEres")
  !HLFIR: %[[EXT_VALUE_ADDR:.*]]:2 = hlfir.declare %[[EXT_VALUE_ARG]]
  !HLFIR: %[[EXT_COND_LOAD:.*]] = fir.load %[[EXT_COND_ADDR]]#0 : !fir.ref<!fir.logical<4>>
  !HLFIR: %[[EXT_NCOND:.*]] = fir.convert %[[EXT_COND_LOAD]] : (!fir.logical<4>) -> i1
  !HLFIR: omp.dispatch nocontext(%[[EXT_NCOND]]) {
  !$omp dispatch nocontext(cond)
  !HLFIR: %[[EXT_VALUE:.*]] = fir.load %[[EXT_VALUE_ADDR]]#0 : !fir.ref<i32>
  !HLFIR-NEXT: %[[EXT_RESULT:.*]] = fir.if %[[EXT_NCOND]] -> (i32) {
  !HLFIR-NEXT: %[[EXT_HOST:.*]] = fir.call @_QPexternal_host_func(%[[EXT_VALUE]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.result %[[EXT_HOST]] : i32
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: %[[EXT_DISPATCH:.*]] = fir.call @_QPexternal_dispatch_func(%[[EXT_VALUE]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.result %[[EXT_DISPATCH]] : i32
  !HLFIR-NEXT: }
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
      !$omp declare variant(argument_base:argument_dispatch) match(construct={dispatch}, user={condition(score(2): .true.)})
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
      !$omp declare variant(target_base:target_dispatch) match(construct={dispatch}, user={condition(score(2): .true.)})
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
  !HLFIR-NOT: fir.if
  !HLFIR: %[[BOTH_ARG_RESULT:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.if %[[ARG_C1]] {
  !HLFIR-NEXT: fir.call @_QPtarget_base(%[[BOTH_ARG_RESULT]]) {{.*}}: (i32) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.if %[[ARG_C2]] {
  !HLFIR-NEXT: fir.call @_QPtarget_host(%[[BOTH_ARG_RESULT]]) {{.*}}: (i32) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QPtarget_dispatch(%[[BOTH_ARG_RESULT]]) {{.*}}: (i32) -> ()
  !HLFIR-NEXT: }
  !HLFIR-NEXT: }
  !HLFIR-NEXT: omp.terminator
  call target_base(argument_base(3))

  !HLFIR: omp.dispatch nocontext(%[[FUNC_C2:.*]]) novariants(%[[FUNC_C1:.*]]) {
  !$omp dispatch novariants(c1) nocontext(c2)
  !HLFIR-NOT: fir.if
  !HLFIR: %[[INNER_RESULT:.*]] = fir.call @_QPargument_host(%{{.*}}) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: %[[OUTER_RESULT:.*]] = fir.if %[[FUNC_C1]] -> (i32) {
  !HLFIR-NEXT: %[[FUNC_BASE:.*]] = fir.call @_QPargument_base(%[[INNER_RESULT]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.result %[[FUNC_BASE]] : i32
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: %[[FUNC_CONTEXT:.*]] = fir.if %[[FUNC_C2]] -> (i32) {
  !HLFIR-NEXT: %[[FUNC_HOST:.*]] = fir.call @_QPargument_host(%[[INNER_RESULT]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.result %[[FUNC_HOST]] : i32
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: %[[FUNC_DISPATCH:.*]] = fir.call @_QPargument_dispatch(%[[INNER_RESULT]]) {{.*}}: (i32) -> i32
  !HLFIR-NEXT: fir.result %[[FUNC_DISPATCH]] : i32
  !HLFIR-NEXT: }
  !HLFIR-NEXT: fir.result %[[FUNC_CONTEXT]] : i32
  !HLFIR-NEXT: }
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
      !$omp declare variant(cast_base:cast_dispatch) match(construct={dispatch}, user={condition(score(2): .true.)})
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
  !HLFIR-NEXT: fir.if %[[CAST_NV]] {
  !HLFIR-NEXT: %[[NV_BASE_ADDR:.*]] = fir.address_of(@_QPcast_base) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NV_BASE:.*]] = fir.convert %[[NV_BASE_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: fir.call %[[NV_BASE]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call %[[NV_DISPATCH]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: }
  !HLFIR-NEXT: omp.terminator
  call cast_base(values)

  !HLFIR: omp.dispatch nocontext(%[[CAST_NC:.*]]) {
  !$omp dispatch nocontext(c2)
  !HLFIR: %[[NC_DISPATCH_ADDR:.*]] = fir.address_of(@_QPcast_dispatch) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NC_DISPATCH:.*]] = fir.convert %[[NC_DISPATCH_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: fir.if %[[CAST_NC]] {
  !HLFIR-NEXT: %[[NC_HOST_ADDR:.*]] = fir.address_of(@_QPcast_host) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[NC_HOST:.*]] = fir.convert %[[NC_HOST_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: fir.call %[[NC_HOST]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call %[[NC_DISPATCH]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: }
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
  !HLFIR-NEXT: fir.if %[[CAST_C1_I1]] {
  !HLFIR-NEXT: %[[BOTH_BASE_ADDR:.*]] = fir.address_of(@_QPcast_base) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[CAST_BASE:.*]] = fir.convert %[[BOTH_BASE_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: fir.call %[[CAST_BASE]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.if %[[CAST_C2_I1]] {
  !HLFIR-NEXT: %[[BOTH_HOST_ADDR:.*]] = fir.address_of(@_QPcast_host) : (!fir.box<!fir.array<?xf32>>) -> ()
  !HLFIR-NEXT: %[[CAST_HOST:.*]] = fir.convert %[[BOTH_HOST_ADDR]] : ((!fir.box<!fir.array<?xf32>>) -> ()) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
  !HLFIR-NEXT: fir.call %[[CAST_HOST]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call %[[CAST_DISPATCH]](%[[VALUES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
  !HLFIR-NEXT: }
  !HLFIR-NEXT: }
  !HLFIR-NEXT: omp.terminator
  call cast_base(values)
end subroutine

! An internal base procedure takes the host association tuple, which an
! external variant does not, so each branch passes its own callee's arguments.
!HLFIR-LABEL: func @_QPtest_host_association(
subroutine test_host_association(c1, c2)
  implicit none
  logical :: c1, c2
  integer :: captured, value
  interface
    subroutine external_value_variant(value)
      integer, intent(out) :: value
    end subroutine
  end interface

  !HLFIR: %[[VALUE:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFtest_host_associationEvalue")
  !HLFIR: %[[TUPLE:.*]] = fir.alloca tuple<!fir.ref<i32>>
  captured = 37

  !HLFIR: omp.dispatch novariants(%[[HA_NV:.*]]) {
  !$omp dispatch novariants(c1)
  !HLFIR-NEXT: fir.if %[[HA_NV]] {
  !HLFIR-NEXT: fir.call @_QFtest_host_associationPinternal_base(%[[VALUE]]#0, %[[TUPLE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QPexternal_value_variant(%[[VALUE]]#0) {{.*}}: (!fir.ref<i32>) -> ()
  !HLFIR-NEXT: }
  call internal_base(value)
  !HLFIR-NEXT: omp.terminator

  !HLFIR: omp.dispatch nocontext(%[[HA_NC:.*]]) {
  !$omp dispatch nocontext(c2)
  !HLFIR-NEXT: fir.if %[[HA_NC]] {
  !HLFIR-NEXT: fir.call @_QFtest_host_associationPinternal_base(%[[VALUE]]#0, %[[TUPLE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QPexternal_value_variant(%[[VALUE]]#0) {{.*}}: (!fir.ref<i32>) -> ()
  !HLFIR-NEXT: }
  call internal_base(value)
  !HLFIR-NEXT: omp.terminator

  ! Both procedures are internal and have the same signature; each is still
  ! called directly with the tuple.
  !HLFIR: omp.dispatch novariants(%[[HB_NV:.*]]) {
  !$omp dispatch novariants(c1)
  !HLFIR-NEXT: fir.if %[[HB_NV]] {
  !HLFIR-NEXT: fir.call @_QFtest_host_associationPinternal_both(%[[VALUE]]#0, %[[TUPLE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.call @_QFtest_host_associationPinternal_variant(%[[VALUE]]#0, %[[TUPLE]]) {{.*}}: (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
  !HLFIR-NEXT: }
  call internal_both(value)
  !HLFIR-NEXT: omp.terminator
contains
  subroutine internal_base(value)
    !$omp declare variant(internal_base:external_value_variant) match(construct={dispatch})
    integer, intent(out) :: value
    value = captured
  end subroutine

  subroutine internal_variant(value)
    integer, intent(out) :: value
    value = captured + 1
  end subroutine

  subroutine internal_both(value)
    !$omp declare variant(internal_both:internal_variant) match(construct={dispatch})
    integer, intent(out) :: value
    value = captured
  end subroutine
end subroutine

! Each array result is saved in its call's branch to the same result storage.
!HLFIR-LABEL: func @_QPtest_dispatch_array_result(
subroutine test_dispatch_array_result(nv, nc, values)
  implicit none
  logical :: nv, nc
  integer :: values(3)
  interface
    function array_dispatch(value) result(output)
      integer, value :: value
      integer :: output(3)
    end function
    function array_user(value) result(output)
      integer, value :: value
      integer :: output(3)
    end function
    function array_base(value) result(output)
      import :: array_dispatch, array_user
      !$omp declare variant(array_base:array_dispatch) match(construct={dispatch})
      !$omp declare variant(array_base:array_user) match(user={condition(.true.)})
      integer, value :: value
      integer :: output(3)
    end function
    integer function array_argument()
    end function
  end interface

  !HLFIR: omp.dispatch nocontext(%[[ARRAY_NC:.*]]) novariants(%[[ARRAY_NV:.*]]) {
  !HLFIR: %[[ARRAY_ARG:.*]] = fir.call @_QParray_argument() {{.*}}: () -> i32
  !HLFIR-NOT: fir.call
  !HLFIR: %[[ARRAY_SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  !HLFIR-NEXT: %[[ARRAY_EXPR:.*]] = hlfir.eval_in_mem shape %[[ARRAY_SHAPE]] : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  !HLFIR-NEXT: ^bb0(%[[ARRAY_STORAGE:[^:]+]]: !fir.ref<!fir.array<3xi32>>):
  !HLFIR-NEXT: fir.if %[[ARRAY_NV]] {
  !HLFIR-NEXT: %[[ARRAY_BASE:.*]] = fir.call @_QParray_base(%[[ARRAY_ARG]]) {{.*}}: (i32) -> !fir.array<3xi32>
  !HLFIR-NEXT: fir.save_result %[[ARRAY_BASE]] to %[[ARRAY_STORAGE]](%[[ARRAY_SHAPE]]) : !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>, !fir.shape<1>
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: fir.if %[[ARRAY_NC]] {
  !HLFIR-NEXT: %[[ARRAY_USER:.*]] = fir.call @_QParray_user(%[[ARRAY_ARG]]) {{.*}}: (i32) -> !fir.array<3xi32>
  !HLFIR-NEXT: fir.save_result %[[ARRAY_USER]] to %[[ARRAY_STORAGE]](%[[ARRAY_SHAPE]]) : !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>, !fir.shape<1>
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: %[[ARRAY_DISPATCH:.*]] = fir.call @_QParray_dispatch(%[[ARRAY_ARG]]) {{.*}}: (i32) -> !fir.array<3xi32>
  !HLFIR-NEXT: fir.save_result %[[ARRAY_DISPATCH]] to %[[ARRAY_STORAGE]](%[[ARRAY_SHAPE]]) : !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>, !fir.shape<1>
  !HLFIR-NEXT: }
  !HLFIR-NEXT: }
  !HLFIR-NEXT: }
  !HLFIR-NEXT: hlfir.assign %[[ARRAY_EXPR]] to %{{.*}} : !hlfir.expr<3xi32>, !fir.ref<!fir.array<3xi32>>
  !HLFIR-NEXT: hlfir.destroy %[[ARRAY_EXPR]] : !hlfir.expr<3xi32>
  !HLFIR-NEXT: omp.terminator
  !$omp dispatch novariants(nv) nocontext(nc)
  values = array_base(array_argument())
end subroutine

module dispatch_result
  implicit none
  type :: pair
    integer :: v(2)
  end type
contains
  function pair_variant() result(r)
    type(pair) :: r
    r%v = 2
  end function
  function pair_base() result(r)
    !$omp declare variant(pair_base:pair_variant) match(construct={dispatch})
    type(pair) :: r
    r%v = 1
  end function
end module

! A result returned in memory is saved in each branch, through the declare
! of the result temporary.
!HLFIR-LABEL: func @_QPtest_result_in_memory(
subroutine test_result_in_memory(cond, x)
  use dispatch_result
  implicit none
  logical :: cond
  type(pair) :: x

  !HLFIR: %[[RES:.*]] = fir.alloca !fir.type<_QMdispatch_resultTpair{{.*}}> <{bindc_name = ".result"}>
  !HLFIR: omp.dispatch novariants(%[[COND:.*]]) {
  !HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[RES]] uniq_name(".tmp.func_result")
  !HLFIR-NEXT: fir.if %[[COND]] {
  !HLFIR-NEXT: %[[BASE:.*]] = fir.call @_QMdispatch_resultPpair_base() {{.*}}
  !HLFIR-NEXT: fir.save_result %[[BASE]] to %[[DECL]]#0
  !HLFIR-NEXT: } else {
  !HLFIR-NEXT: %[[VARIANT:.*]] = fir.call @_QMdispatch_resultPpair_variant() {{.*}}
  !HLFIR-NEXT: fir.save_result %[[VARIANT]] to %[[DECL]]#0
  !HLFIR-NEXT: {{^ *[}]$}}
  !$omp dispatch novariants(cond)
  x = pair_base()
end subroutine

!HLFIR-DAG: func.func private @_QPexternal_variant()
!HLFIR-DAG: func.func private @_QPexternal_base()
!HLFIR-DAG: func.func private @_QPexternal_dispatch_func(i32) -> i32
!HLFIR-DAG: func.func private @_QPexternal_host_func(i32) -> i32
