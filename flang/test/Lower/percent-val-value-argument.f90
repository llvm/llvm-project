! RUN: flang -fc1 -emit-hlfir %s -o - | FileCheck %s

program main
  logical::a1
  data a1/.true./
  call sb(%val(a1))
! CHECK: %[[A1_ADDR:.*]] = fir.address_of(@_QFEa1) : !fir.ref<!fir.logical<4>>
! CHECK: %[[A1_DECL:.*]]:2 = hlfir.declare %[[A1_ADDR]] {uniq_name = "_QFEa1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK: %[[A1_LOADED:.*]] = fir.load %[[A1_DECL]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: fir.call @_QFPsb(%[[A1_LOADED]]) fastmath<contract> : (!fir.logical<4>) -> ()
! CHECK: func.func private @_QFPsb(%[[SB_ARG:.*]]: !fir.logical<4> {fir.bindc_name = "x1"})
  write(6,*) "a1 = ", a1
contains
  subroutine sb(x1)
    logical, value :: x1
  end subroutine sb
end program main
