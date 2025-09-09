! RUN: flang -fc1 -emit-hlfir %s -o - | FileCheck %s

program main
  logical::a1
  data a1/.true./
  call sa(%val(a1))
! CHECK: %[[A1_ADDR:.*]] = fir.address_of(@_QFEa1) : !fir.ref<!fir.logical<4>>
! CHECK: %[[A1_DECL:.*]]:2 = hlfir.declare %[[A1_ADDR]] {uniq_name = "_QFEa1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: fir.call @_QPsa(%[[A1_DECL]]#0) fastmath<contract> : (!fir.ref<!fir.logical<4>>) -> ()
! CHECK: func.func @_QPsa(%[[SA_ARG:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x1"}) {
  write(6,*) "a1 = ", a1
end program main

subroutine sa(x1)
  logical::x1
end subroutine sa
