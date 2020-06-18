! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test dummy procedures

! Test of dummy procedure call
! CHECK-LABEL: func @_QPfoo(%arg0: () -> ()) -> f32
real function foo(bar)
  real :: bar, x
  ! CHECK: %[[x:.*]] = fir.alloca f32 {name = "x"}
  x = 42.
  ! CHECK: %[[funccast:.*]] = fir.convert %arg0 : (() -> ()) -> ((!fir.ref<f32>) -> f32)
  ! CHECK: fir.call %[[funccast]](%[[x]]) : (!fir.ref<f32>) -> f32
  foo = bar(x)
end function

! Test case where dummy procedure is only transiting.
! CHECK-LABEL: func @_QPprefoo(%arg0: () -> ()) -> f32
real function prefoo(bar)
  external :: bar
  ! CHECK: fir.call @_QPfoo(%arg0) : (() -> ()) -> f32
  prefoo = foo(bar)
end function

! Function that will be passed as dummy argument
!CHECK-LABEL: func @_QPfunc(%arg0: !fir.ref<f32>) -> f32
real function func(x)
  real :: x
  func = x + 0.5
end function

! Test passing functions as dummy procedure arguments
! CHECK-LABEL: func @_QPtest_func
real function test_func()
  real :: func, prefoo
  external :: func
  !CHECK: %[[f:.*]] = constant @_QPfunc : (!fir.ref<f32>) -> f32
  !CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.ref<f32>) -> f32) -> (() -> ())
  !CHECK: fir.call @_QPprefoo(%[[fcast]]) : (() -> ()) -> f32
  test_func = prefoo(func)
end function

! Repeat test with dummy subroutine

! CHECK-LABEL: func @_QPfoo_sub(%arg0: () -> ())
subroutine foo_sub(bar_sub)
  ! CHECK: %[[x:.*]] = fir.alloca f32 {name = "x"}
  x = 42.
  ! CHECK: %[[funccast:.*]] = fir.convert %arg0 : (() -> ()) -> ((!fir.ref<f32>) -> ())
  ! CHECK: fir.call %[[funccast]](%[[x]]) : (!fir.ref<f32>)
  call bar_sub(x)
end subroutine

! Test case where dummy procedure is only transiting.
! CHECK-LABEL: func @_QPprefoo_sub(%arg0: () -> ())
subroutine prefoo_sub(bar_sub)
  external :: bar_sub
  ! CHECK: fir.call @_QPfoo_sub(%arg0) : (() -> ()) -> ()
  call foo_sub(bar_sub)
end subroutine

! Subroutine that will be passed as dummy argument
!CHECK-LABEL: func @_QPsub(%arg0: !fir.ref<f32>)
subroutine sub(x)
  real :: x
  print *, x
end subroutine

! Test passing functions as dummy procedure arguments
! CHECK-LABEL: func @_QPtest_sub
subroutine test_sub()
  external :: sub
  !CHECK: %[[f:.*]] = constant @_QPsub : (!fir.ref<f32>) -> ()
  !CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.ref<f32>) -> ()) -> (() -> ())
  !CHECK: fir.call @_QPprefoo_sub(%[[fcast]]) : (() -> ()) -> ()
  call prefoo_sub(sub)
end subroutine

! FIXME: create funcOp if not defined in file
!subroutine todo1()
!  external proc_not_defined_in_file
!  call prefoo_sub(proc_not_defined_in_file)
!end subroutine

! FIXME: pass intrinsics
!subroutine todo2()
!  intrinsic :: acos
!  print *, prefoo(acos)
!end subroutine

! TODO: improve dummy procedure types when interface is given.
! CHECK: func @_QPtodo3(%arg0: () -> ())
! SHOULD-CHECK: func @_QPtodo3(%arg0: (!fir.ref<f32>) -> f32)
subroutine todo3(dummy_proc)
  intrinsic :: acos
  procedure(acos) :: dummy_proc
end subroutine
