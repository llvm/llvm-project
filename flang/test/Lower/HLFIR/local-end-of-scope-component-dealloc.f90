! Test automatic deallocation of allocatable components
! of local variables as described in Fortran 2018 standard
! 9.7.3.2 point 2. and 3.
! The allocatable components of local variables are local variables
! themselves due to 5.4.3.2.2 p. 2, note 1.
! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

module types
  type t1
     real, allocatable :: x
  end type t1
  type t2
     type(t1) :: x
  end type t2
  type, extends(t1) :: t3
  end type t3
  type, extends(t3) :: t4
  end type t4
  type, extends(t2) :: t5
  end type t5
end module types

subroutine test1()
  use types
  type(t1) :: x1
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_9:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_9]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>) -> !fir.box<none>

subroutine test1b()
  use types
  block
    type(t1) :: x1
  end block
end subroutine test1b
! CHECK-LABEL:   func.func @_QPtest1b() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_10:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_10]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>) -> !fir.box<none>

subroutine test2()
  use types
  type(t2) :: x2
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_9:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_9]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt2{x:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>>) -> !fir.box<none>

subroutine test2b()
  use types
  block
    type(t2) :: x2
  end block
end subroutine test2b
! CHECK-LABEL:   func.func @_QPtest2b() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_10:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_10]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt2{x:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>>) -> !fir.box<none>

subroutine test3()
  use types
  type(t3) :: x3
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_9:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_9]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt3{t1:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>>) -> !fir.box<none>

subroutine test3b()
  use types
  block
    type(t3) :: x3
  end block
end subroutine test3b
! CHECK-LABEL:   func.func @_QPtest3b() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_10:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_10]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt3{t1:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>>) -> !fir.box<none>

subroutine test4()
  use types
  type(t4) :: x4
end subroutine test4
! CHECK-LABEL:   func.func @_QPtest4() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_9:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_9]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt4{t3:!fir.type<_QMtypesTt3{t1:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.box<none>

subroutine test4b()
  use types
  block
    type(t4) :: x4
  end block
end subroutine test4b
! CHECK-LABEL:   func.func @_QPtest4b() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_10:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_10]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt4{t3:!fir.type<_QMtypesTt3{t1:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.box<none>

subroutine test5()
  use types
  type(t5) :: x5
end subroutine test5
! CHECK-LABEL:   func.func @_QPtest5() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_9:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_9]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt5{t2:!fir.type<_QMtypesTt2{x:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.box<none>

subroutine test5b()
  use types
  block
    type(t5) :: x5
  end block
end subroutine test5b
! CHECK-LABEL:   func.func @_QPtest5b() {
! CHECK-DAG:       fir.call @_FortranADestroy(%[[VAL_10:.*]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK-DAG:       %[[VAL_10]] = fir.convert %{{.*}} : (!fir.box<!fir.type<_QMtypesTt5{t2:!fir.type<_QMtypesTt2{x:!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>}>}>>) -> !fir.box<none>
