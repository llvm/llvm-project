! Test that the hlfir.end_associate generated for the argument
! passing has operand that is a Fortran entity, so that
! the shape/type-params information is available
! during bufferization that will have to generate a runtime call
! for deallocating the allocatable component of the temporary.
!
! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

module types
  type t
     real, allocatable :: x
  end type t
contains
end module types

subroutine test1(x)
  use types
  interface
     subroutine callee1(x)
       use types
       type(t), value :: x(10)
     end subroutine callee1
  end interface
  type(t) :: x(:)
  call callee1(x)
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %{{.*}}(%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1)
! CHECK:           hlfir.end_associate %[[VAL_6]]#0, %[[VAL_6]]#2 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1

subroutine test2(x)
  use types
  interface
     subroutine callee2(x)
       use types
       type(t) :: x(:)
     end subroutine callee2
  end interface
  type(t) :: x(:)
  call callee2((x))
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %{{.*}}(%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1)
! CHECK:           hlfir.end_associate %[[VAL_9]]#0, %[[VAL_9]]#2 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1

subroutine test3(x)
  use types
  interface
     subroutine callee3(x)
       use types
       type(t), optional, value :: x(10)
     end subroutine callee3
  end interface
  type(t), optional :: x(:)
  call callee3(x)
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK:           %[[VAL_3:.*]]:3 = fir.if %{{.*}} -> (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1) {
! CHECK:             %[[VAL_8:.*]]:3 = hlfir.associate %{{.*}}(%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1)
! CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>
! CHECK:             fir.result %[[VAL_9]], %[[VAL_8]]#0, %[[VAL_8]]#2 : !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1
! CHECK:           } else {
! CHECK:             fir.result %{{.*}}, %{{.*}}, %{{.*}} : !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1
! CHECK:           }
! CHECK:           hlfir.end_associate %[[VAL_3]]#1, %[[VAL_3]]#2 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1
