! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test passing allocatables on caller side

! CHECK-LABEL: func @_QPtest_scalar_call(
subroutine test_scalar_call()
  interface
  subroutine test_scalar(x)
    real, allocatable :: x
  end subroutine
  end interface
  real, allocatable :: x
  ! CHECK: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {{{.*}}uniq_name = "_QFtest_scalar_callEx"}
  call test_scalar(x)
  ! CHECK: fir.call @_QPtest_scalar(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_array_call(
subroutine test_array_call()
  interface
  subroutine test_array(x)
    integer, allocatable :: x(:)
  end subroutine
  end interface
  integer, allocatable :: x(:)
  ! CHECK: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {{{.*}}uniq_name = "_QFtest_array_callEx"}
  call test_array(x)
  ! CHECK: fir.call @_QPtest_array(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_scalar_deferred_call(
subroutine test_char_scalar_deferred_call()
  interface
  subroutine test_char_scalar_deferred(x)
    character(:), allocatable :: x
  end subroutine
  end interface
  character(:), allocatable :: x
  ! CHECK: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFtest_char_scalar_deferred_callEx"}
  call test_char_scalar_deferred(x)
  ! CHECK: fir.call @_QPtest_char_scalar_deferred(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_scalar_explicit_call(
subroutine test_char_scalar_explicit_call(n)
  integer :: n
  interface
  subroutine test_char_scalar_explicit(x)
    character(10), allocatable :: x
  end subroutine
  end interface
  character(10), allocatable :: x
  character(n), allocatable :: x2
  ! CHECK-DAG: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {{{.*}}uniq_name = "_QFtest_char_scalar_explicit_callEx"}
  ! CHECK-DAG: %[[box2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFtest_char_scalar_explicit_callEx2"}
  call test_char_scalar_explicit(x)
  ! CHECK: fir.call @_QPtest_char_scalar_explicit(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> ()
  call test_char_scalar_explicit(x2)
  ! CHECK: %[[box2cast:.*]] = fir.convert %[[box2]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  ! CHECK: fir.call @_QPtest_char_scalar_explicit(%[[box2cast]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_array_deferred_call(
subroutine test_char_array_deferred_call()
  interface
  subroutine test_char_array_deferred(x)
    character(:), allocatable :: x(:)
  end subroutine
  end interface
  character(:), allocatable :: x(:)
  ! CHECK: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFtest_char_array_deferred_callEx"}
  call test_char_array_deferred(x)
  ! CHECK: fir.call @_QPtest_char_array_deferred(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_char_array_explicit_call(
subroutine test_char_array_explicit_call(n)
  integer :: n
  interface
  subroutine test_char_array_explicit(x)
    character(10), allocatable :: x(:)
  end subroutine
  end interface
  character(10), allocatable :: x(:)
  character(n), allocatable :: x2(:)
  ! CHECK-DAG: %[[box:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = "_QFtest_char_array_explicit_callEx"}
  ! CHECK-DAG: %[[box2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = "_QFtest_char_array_explicit_callEx2"}
  call test_char_array_explicit(x)
  ! CHECK: fir.call @_QPtest_char_array_explicit(%[[box]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> ()
  call test_char_array_explicit(x2)
  ! CHECK: %[[box2cast:.*]] = fir.convert %[[box2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: fir.call @_QPtest_char_array_explicit(%[[box2cast]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> ()
end subroutine
