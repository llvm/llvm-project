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

! Test passing unrestricted intrinsics

! Intrinsic using runtime
! CHECK-LABEL: func @_QPtest_acos
subroutine test_acos(x)
  intrinsic :: acos
  !CHECK: %[[f:.*]] = constant @fir.acos.f32.ref_f32 : (!fir.ref<f32>) -> f32
  !CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.ref<f32>) -> f32) -> (() -> ())
  !CHECK: fir.call @_QPfoo_acos(%[[fcast]]) : (() -> ()) -> ()
  call foo_acos(acos)
end subroutine

! Intrinsic implemented inlined
! CHECK-LABEL: func @_QPtest_aimag
subroutine test_aimag()
  intrinsic :: aimag
  !CHECK: %[[f:.*]] = constant @fir.aimag.f32.ref_z4 : (!fir.ref<!fir.complex<4>>) -> f32
  !CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.ref<!fir.complex<4>>) -> f32) -> (() -> ())
  !CHECK: fir.call @_QPfoo_aimag(%[[fcast]]) : (() -> ()) -> ()
  call foo_aimag(aimag)
end subroutine

! Character Intrinsic implemented inlined
! CHECK-LABEL: func @_QPtest_len
subroutine test_len()
  intrinsic :: len
  ! CHECK: %[[f:.*]] = constant @fir.len.i32.bc1 : (!fir.boxchar<1>) -> i32
  ! CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.boxchar<1>) -> i32) -> (() -> ())
  !CHECK: fir.call @_QPfoo_len(%[[fcast]]) : (() -> ()) -> ()
  call foo_len(len)
end subroutine


! Intrinsic implemented inlined with specific name different from generic
! CHECK-LABEL: func @_QPtest_iabs
subroutine test_iabs()
  intrinsic :: iabs
  ! CHECK: %[[f:.*]] = constant @fir.abs.i32.ref_i32 : (!fir.ref<i32>) -> i32
  ! CHECK: %[[fcast:.*]] = fir.convert %f : ((!fir.ref<i32>) -> i32) -> (() -> ())
  ! CHECK: fir.call @_QPfoo_iabs(%[[fcast]]) : (() -> ()) -> ()
  call foo_iabs(iabs)
end subroutine


! TODO: exhaustive test of unrestricted intrinsic table 16.2 

! FIXME: create funcOp if not defined in file
!subroutine todo1()
!  external proc_not_defined_in_file
!  call prefoo_sub(proc_not_defined_in_file)
!end subroutine

! TODO: improve dummy procedure types when interface is given.
! CHECK: func @_QPtodo3(%arg0: () -> ())
! SHOULD-CHECK: func @_QPtodo3(%arg0: (!fir.ref<f32>) -> f32)
subroutine todo3(dummy_proc)
  intrinsic :: acos
  procedure(acos) :: dummy_proc
end subroutine

! CHECK-LABEL: func @fir.acos.f32.ref_f32(%arg0: !fir.ref<f32>) -> f32
  !CHECK: %[[load:.*]] = fir.load %arg0
  !CHECK: %[[res:.*]] = call @__fs_acos_1(%[[load]]) : (f32) -> f32
  !CHECK: return %[[res]] : f32

!CHECK-LABEL: func @fir.aimag.f32.ref_z4(%arg0: !fir.ref<!fir.complex<4>>)
  !CHECK: %[[load:.*]] = fir.load %arg0
  !CHECK: %[[cst1:.*]] = constant 1
  !CHECK: %[[imag:.*]] = fir.extract_value %[[load]], %[[cst1]] : (!fir.complex<4>, index) -> f32
  !CHECK: return %[[imag]] : f32

!CHECK-LABEL: func @fir.len.i32.bc1(%arg0: !fir.boxchar<1>)
  !CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, index)
  !CHECK: %[[len:.*]] = fir.convert %[[unboxed]]#1 : (index) -> i32
  !CHECK: return %[[len]] : i32
