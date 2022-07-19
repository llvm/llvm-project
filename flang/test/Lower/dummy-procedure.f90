! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test dummy procedures

! Test of dummy procedure call
! CHECK-LABEL: func @_QPfoo(
! CHECK-SAME: %{{.*}}: !fir.boxproc<() -> ()>{{.*}}) -> f32
real function foo(bar)
  real :: bar, x
  ! CHECK: %[[x:.*]] = fir.alloca f32 {{{.*}}uniq_name = "{{.*}}Ex"}
  x = 42.
  ! CHECK: %[[funccast:.*]] = fir.box_addr %arg0 : (!fir.boxproc<() -> ()>) -> ((!fir.ref<f32>) -> f32)
  ! CHECK: fir.call %[[funccast]](%[[x]]) : (!fir.ref<f32>) -> f32
  foo = bar(x)
end function

! Test case where dummy procedure is only transiting.
! CHECK-LABEL: func @_QPprefoo(
! CHECK-SAME: %{{.*}}: !fir.boxproc<() -> ()>{{.*}}) -> f32
real function prefoo(bar)
  external :: bar
  ! CHECK: fir.call @_QPfoo(%arg0) : (!fir.boxproc<() -> ()>) -> f32
  prefoo = foo(bar)
end function

! Function that will be passed as dummy argument
! CHECK-LABEL: func @_QPfunc(
! CHECK-SAME: %{{.*}}: !fir.ref<f32>{{.*}}) -> f32
real function func(x)
  real :: x
  func = x + 0.5
end function

! Test passing functions as dummy procedure arguments
! CHECK-LABEL: func @_QPtest_func
real function test_func()
  real :: func, prefoo
  external :: func
  !CHECK: %[[f:.*]] = fir.address_of(@_QPfunc) : (!fir.ref<f32>) -> f32
  !CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
  !CHECK: fir.call @_QPprefoo(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> f32
  test_func = prefoo(func)
end function

! Repeat test with dummy subroutine

! CHECK-LABEL: func @_QPfoo_sub(
! CHECK-SAME: %{{.*}}: !fir.boxproc<() -> ()>{{.*}})
subroutine foo_sub(bar_sub)
  ! CHECK: %[[x:.*]] = fir.alloca f32 {{{.*}}uniq_name = "{{.*}}Ex"}
  x = 42.
  ! CHECK: %[[funccast:.*]] = fir.box_addr %arg0 : (!fir.boxproc<() -> ()>) -> ((!fir.ref<f32>) -> ())
  ! CHECK: fir.call %[[funccast]](%[[x]]) : (!fir.ref<f32>)
  call bar_sub(x)
end subroutine

! Test case where dummy procedure is only transiting.
! CHECK-LABEL: func @_QPprefoo_sub(
! CHECK-SAME: %{{.*}}: !fir.boxproc<() -> ()>{{.*}})
subroutine prefoo_sub(bar_sub)
  external :: bar_sub
  ! CHECK: fir.call @_QPfoo_sub(%arg0) : (!fir.boxproc<() -> ()>) -> ()
  call foo_sub(bar_sub)
end subroutine

! Subroutine that will be passed as dummy argument
! CHECK-LABEL: func @_QPsub(
! CHECK-SAME: %{{.*}}: !fir.ref<f32>{{.*}})
subroutine sub(x)
  real :: x
  print *, x
end subroutine

! Test passing functions as dummy procedure arguments
! CHECK-LABEL: func @_QPtest_sub
subroutine test_sub()
  external :: sub
  !CHECK: %[[f:.*]] = fir.address_of(@_QPsub) : (!fir.ref<f32>) -> ()
  !CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<f32>) -> ()) -> !fir.boxproc<() -> ()>
  !CHECK: fir.call @_QPprefoo_sub(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call prefoo_sub(sub)
end subroutine

! CHECK-LABEL: func @_QPpassing_not_defined_in_file()
subroutine passing_not_defined_in_file()
  external proc_not_defined_in_file
  ! CHECK: %[[addr:.*]] = fir.address_of(@_QPproc_not_defined_in_file) : () -> ()
  ! CHECK: %[[ep:.*]] = fir.emboxproc %[[addr]]
  ! CHECK: fir.call @_QPprefoo_sub(%[[ep]]) : (!fir.boxproc<() -> ()>) -> ()
  call prefoo_sub(proc_not_defined_in_file)
end subroutine

! Test passing unrestricted intrinsics

! Intrinsic using runtime
! CHECK-LABEL: func @_QPtest_acos
subroutine test_acos(x)
  intrinsic :: acos
  !CHECK: %[[f:.*]] = fir.address_of(@fir.acos.f32.ref_f32) : (!fir.ref<f32>) -> f32
  !CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
  !CHECK: fir.call @_QPfoo_acos(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call foo_acos(acos)
end subroutine

! CHECK-LABEL: func @_QPtest_atan2
subroutine test_atan2()
  intrinsic :: atan2
  ! CHECK: %[[f:.*]] = fir.address_of(@fir.atan2.f32.ref_f32.ref_f32) : (!fir.ref<f32>, !fir.ref<f32>) -> f32
  ! CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<f32>, !fir.ref<f32>) -> f32) -> !fir.boxproc<() -> ()>
  ! CHECK: fir.call @_QPfoo_atan2(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call foo_atan2(atan2)
end subroutine

! Intrinsic implemented inlined
! CHECK-LABEL: func @_QPtest_aimag
subroutine test_aimag()
  intrinsic :: aimag
  !CHECK: %[[f:.*]] = fir.address_of(@fir.aimag.f32.ref_z4) : (!fir.ref<!fir.complex<4>>) -> f32
  !CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<!fir.complex<4>>) -> f32) -> !fir.boxproc<() -> ()>
  !CHECK: fir.call @_QPfoo_aimag(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call foo_aimag(aimag)
end subroutine

! Character Intrinsic implemented inlined
! CHECK-LABEL: func @_QPtest_len
subroutine test_len()
  intrinsic :: len
  ! CHECK: %[[f:.*]] = fir.address_of(@fir.len.i32.bc1) : (!fir.boxchar<1>) -> i32
  ! CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.boxchar<1>) -> i32) -> !fir.boxproc<() -> ()>
  !CHECK: fir.call @_QPfoo_len(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call foo_len(len)
end subroutine

! Intrinsic implemented inlined with specific name different from generic
! CHECK-LABEL: func @_QPtest_iabs
subroutine test_iabs()
  intrinsic :: iabs
  ! CHECK: %[[f:.*]] = fir.address_of(@fir.abs.i32.ref_i32) : (!fir.ref<i32>) -> i32
  ! CHECK: %[[fcast:.*]] = fir.emboxproc %[[f]] : ((!fir.ref<i32>) -> i32) -> !fir.boxproc<() -> ()>
  ! CHECK: fir.call @_QPfoo_iabs(%[[fcast]]) : (!fir.boxproc<() -> ()>) -> ()
  call foo_iabs(iabs)
end subroutine

! TODO: exhaustive test of unrestricted intrinsic table 16.2 

! TODO: improve dummy procedure types when interface is given.
! CHECK: func @_QPtodo3(
! CHECK-SAME: %{{.*}}: !fir.boxproc<() -> ()>{{.*}})
! SHOULD-CHECK: func @_QPtodo3(%arg0: (!fir.ref<f32>) -> f32)
subroutine todo3(dummy_proc)
  intrinsic :: acos
  procedure(acos) :: dummy_proc
end subroutine

! CHECK-LABEL: func private @fir.acos.f32.ref_f32(%arg0: !fir.ref<f32>) -> f32
  !CHECK: %[[load:.*]] = fir.load %arg0
  !CHECK: %[[res:.*]] = fir.call @__fs_acos_1(%[[load]]) : (f32) -> f32
  !CHECK: return %[[res]] : f32

! CHECK-LABEL: func private @fir.atan2.f32.ref_f32.ref_f32(
! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>, %[[y:.*]]: !fir.ref<f32>) -> f32
  ! CHECK-DAG: %[[xload:.*]] = fir.load %[[x]] : !fir.ref<f32>
  ! CHECK-DAG: %[[yload:.*]] = fir.load %[[y]] : !fir.ref<f32>
  ! CHECK: %[[atan2:.*]] = math.atan2 %[[xload]], %[[yload]] : f32
  ! CHECK: return %[[atan2]] : f32

!CHECK-LABEL: func private @fir.aimag.f32.ref_z4(%arg0: !fir.ref<!fir.complex<4>>)
  !CHECK: %[[load:.*]] = fir.load %arg0
  !CHECK: %[[imag:.*]] = fir.extract_value %[[load]], [1 : index] : (!fir.complex<4>) -> f32
  !CHECK: return %[[imag]] : f32

!CHECK-LABEL: func private @fir.len.i32.bc1(%arg0: !fir.boxchar<1>)
  !CHECK: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  !CHECK: %[[len:.*]] = fir.convert %[[unboxed]]#1 : (index) -> i32
  !CHECK: return %[[len]] : i32
