! This is a crazy program, recursive derived types with recursive kind
! parameters are a terrible idea if they do not converge quickly.

! RUN: bbc -emit-hlfir -o - -I nw %s | FileCheck %s

subroutine foo(x)
  type t(k)
    integer, kind :: k
    type(t(modulo(k+1,2))), pointer :: p
  end type
  type(t(1)) :: x
end subroutine
! CHECK-LABEL: func.func @_QPfoo(
! CHECK-SAME: !fir.ref<!fir.type<_QFfooTtK1{p:!fir.box<!fir.ptr<!fir.type<_QFfooTtK0{p:!fir.box<!fir.ptr<!fir.type<_QFfooTtK1>>>}>>>}>>
