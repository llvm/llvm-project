! RUN: bbc %s -o - | FileCheck %s

! CHECK: func @_QPtest1(
subroutine test1
  integer i
  ! CHECK-DAG: %[[i:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK-DAG: %[[tup:.*]] = fir.alloca tuple<!fir.ptr<i32>>
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: %[[ii:.*]] = fir.convert %[[i]]
  ! CHECK: fir.store %[[ii]] to %[[addr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: fir.call @_QFtest1Ptest1_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ptr<i32>>>) -> ()
  call test1_internal
  print *, i
contains
  ! CHECK: func @_QFtest1Ptest1_internal(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<i32>>> {fir.host_assoc}) {
  ! CHECK: %[[iaddr:.*]] = fir.coordinate_of %[[arg]], %c0
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.ref<!fir.ptr<i32>>
  ! CHECK: %[[val:.*]] = fir.call @_QPifoo() : () -> i32
  ! CHECK: fir.store %[[val]] to %[[i]] : !fir.ptr<i32>
  subroutine test1_internal
    i = ifoo()
  end subroutine test1_internal
end subroutine test1

! CHECK: func @_QPtest2() {
subroutine test2
  a = 1.0
  b = 2.0
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ptr<f32>, !fir.ptr<f32>>
  ! CHECK-DAG: %[[a0:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK-DAG: %[[p0:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ptr<f32>
  ! CHECK: fir.store %[[p0]] to %[[a0]] : !fir.ref<!fir.ptr<f32>>
  ! CHECK-DAG: %[[b0:.*]] = fir.coordinate_of %[[tup]], %c1
  ! CHECK-DAG: %[[p1:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ptr<f32>
  ! CHECK: fir.store %[[p1]] to %[[b0]] : !fir.ref<!fir.ptr<f32>>
  ! CHECK: fir.call @_QFtest2Ptest2_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>>) -> ()
  call test2_internal
  print *, a, b
contains
  ! CHECK: func @_QFtest2Ptest2_internal(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>> {fir.host_assoc}) {
  subroutine test2_internal
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %{{.*}} = fir.load %[[b]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: fir.alloca
    ! CHECK: fir.load %[[aa]] : !fir.ptr<f32>
    c = a
    a = b
    b = c
    call test2_inner
  end subroutine test2_internal

  ! CHECK: func @_QFtest2Ptest2_inner(%[[arg:[^:]*]]: !fir.ref<tuple<!fir.ptr<f32>, !fir.ptr<f32>>> {fir.host_assoc}) {
  subroutine test2_inner
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %[[bb:.*]] = fir.load %[[b]] : !fir.ref<!fir.ptr<f32>>
    ! CHECK-DAG: %[[bd:.*]] = fir.load %[[bb]] : !fir.ptr<f32>
    ! CHECK-DAG: %[[ad:.*]] = fir.load %[[aa]] : !fir.ptr<f32>
    ! CHECK: %{{.*}} = cmpf ogt, %[[ad]], %[[bd]] : f32
    if (a > b) then
       b = b + 2.0
    end if
  end subroutine test2_inner
end subroutine test2
