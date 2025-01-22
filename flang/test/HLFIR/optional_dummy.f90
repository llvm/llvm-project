! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Check that the lowering does not generate fir.box_addr for
! the optional box. It will cause segfault during execution.

! CHECK-LABEL:   func.func @_QPtest(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "ext_buf", fir.contiguous, fir.optional}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<contiguous, optional>, uniq_name = "_QFtestEext_buf"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#1 : (!fir.box<!fir.array<?xi32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_4:.*]] = arith.constant false
! CHECK:           %[[VAL_5:.*]] = arith.constant false
! CHECK:           fir.call @_FortranAStopStatement(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) fastmath<contract> : (i32, i1, i1) -> ()
! CHECK:           fir.unreachable
! CHECK:         ^bb2:
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }
subroutine test(ext_buf)
  integer, contiguous, optional :: ext_buf(:)
  if (present(ext_buf)) then
     stop
  endif
  return
end subroutine test
