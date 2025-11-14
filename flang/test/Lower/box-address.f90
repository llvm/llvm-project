! RUN: flang -fc1 -emit-hlfir %s -o - | FileCheck %s

module m3
  type x1
     integer::ix1
  end type x1
  type,extends(x1)::x2
  end type x2
  type,extends(x2)::x3
  end type x3
  class(x1),pointer,dimension(:)::cy1
contains
  subroutine dummy()
  entry      chk(c1)
   class(x1),dimension(3)::c1
 end subroutine dummy
end module m3
! CHECK-LABEL: func.func @_QMm3Pchk(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.class<!fir.array<3x!fir.type<_QMm3Tx1{ix1:i32}>>> {fir.bindc_name = "c1"}) {
! CHECK:         %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[DECLARE:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE]] {{.*}} {uniq_name = "_QMm3FdummyEc1"} : (!fir.class<!fir.array<3x!fir.type<_QMm3Tx1{ix1:i32}>>>, !fir.dscope) -> (!fir.class<!fir.array<3x!fir.type<_QMm3Tx1{ix1:i32}>>>, !fir.class<!fir.array<3x!fir.type<_QMm3Tx1{ix1:i32}>>>)

subroutine s1
  use m3
  type(x1),target::ty1(3)
  ty1%ix1=[1,2,3]
  cy1=>ty1
  call chk(cy1)
end subroutine s1

program main
  call s1
  print *,'pass'
end program main
