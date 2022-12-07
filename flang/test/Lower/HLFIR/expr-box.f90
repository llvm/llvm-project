! Test lowering of of expressions as fir.box
! RUN: bbc -hlfir -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>
subroutine foo(x)
  integer :: x(21:30)
  print *, x 
! CHECK-DAG:  %[[VAL_3:.*]] = arith.constant 21 : index
! CHECK-DAG:  %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFfooEx"} : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:  fir.embox %[[VAL_6]]#1(%[[VAL_5]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xi32>>
end subroutine
