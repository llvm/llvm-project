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

subroutine test_place_in_memory_and_embox()
 logical(8) :: a, b
 write(06), a.and.b
end subroutine
! CHECK-LABEL: func.func @_QPtest_place_in_memory_and_embox(
! CHECK:  %[[TEMP:.*]] = fir.alloca !fir.logical<8>
! CHECK:  %[[AND:.*]] = arith.andi {{.*}}
! CHECK:  %[[CAST:.*]] = fir.convert %[[AND]] : (i1) -> !fir.logical<8>
! CHECK:  fir.store %[[CAST]] to %[[TEMP]] : !fir.ref<!fir.logical<8>>
! CHECK:  %[[BOX:.*]] = fir.embox %[[TEMP]] : (!fir.ref<!fir.logical<8>>) -> !fir.box<!fir.logical<8>>
! CHECK:  %[[BOX_CAST:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.logical<8>>) -> !fir.box<none>
! CHECK:  fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_CAST]])

! check we can box a trivial value
subroutine productMask(s, a)
  integer :: s
  integer :: a(:)
  s = product(a, mask=.true.)
endsubroutine
! CHECK-LABEL: func.func @_QPproductmask(
! CHECK:      %[[TRUE:.*]] = arith.constant true
! CHECK:      %[[ALLOC:.*]] = fir.alloca !fir.logical<4>
! CHECK:      %[[TRUE_L4:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
! CHECK-NEXT: fir.store %[[TRUE_L4]] to %[[ALLOC]]
! CHECK-NEXT: %[[BOX:.*]] = fir.embox %[[ALLOC]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
