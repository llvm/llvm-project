! Test passing rank 2 CLASS(*) deferred shape to assumed size assumed type
! This requires copy-in/copy-out logic.
! RUN: bbc -emit-hlfir -polymorphic-type -o - %s | FileCheck %s

subroutine pass_poly_to_assumed_type_assumed_size(x)
  class(*), target :: x(:,:)
  interface
    subroutine assumed_type_assumed_size(x)
       type(*), target :: x(*)
    end subroutine
  end interface
  call assumed_type_assumed_size(x)
end subroutine
! CHECK-LABEL:   func.func @_QPpass_poly_to_assumed_type_assumed_size(
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFpass_poly_to_assumed_type_assumed_sizeEx"} : (!fir.class<!fir.array<?x?xnone>>) -> (!fir.class<!fir.array<?x?xnone>>, !fir.class<!fir.array<?x?xnone>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 : (!fir.class<!fir.array<?x?xnone>>) -> (!fir.class<!fir.array<?x?xnone>>, i1)
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.class<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?x?xnone>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<?x?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:           fir.call @_QPassumed_type_assumed_size(%[[VAL_4]]) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           hlfir.copy_out %[[VAL_2]]#0, %[[VAL_2]]#1 to %[[VAL_1]]#0 : (!fir.class<!fir.array<?x?xnone>>, i1, !fir.class<!fir.array<?x?xnone>>) -> ()
