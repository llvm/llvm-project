! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine assumed_size_test(a)
  integer :: a(10,*)
  a(:, 1:2) = a(:, 3:4)
end subroutine assumed_size_test

! CHECK-LABEL: func.func @_QPassumed_size_test(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.array<10x?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[C10_I64:.*]] = arith.constant 10 : i64
! CHECK:         %[[IDX10:.*]] = fir.convert %[[C10_I64]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index
! CHECK:         %[[COND:.*]] = arith.cmpi sgt, %[[IDX10]], %[[C0]] : index
! CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[IDX10]], %[[C0]] : index
! CHECK:         %[[EXT:.*]] = fir.assumed_size_extent : index
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[SEL]], %[[EXT]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%[[SHAPE]]) dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFassumed_size_testEa"} : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<10x?xi32>>, !fir.ref<!fir.array<10x?xi32>>)
! CHECK:         %[[C1:.*]] = arith.constant 1 : index
! CHECK:         %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:         %[[C10:.*]] = arith.constant 10 : index
! CHECK:         %[[C3:.*]] = arith.constant 3 : index
! CHECK:         %[[C4:.*]] = arith.constant 4 : index
! CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
! CHECK:         %[[C2:.*]] = arith.constant 2 : index
! CHECK:         %[[SHAPE2:.*]] = fir.shape %[[C10]], %[[C2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[SRC:.*]] = hlfir.designate %[[DECL]]#0 (%[[C1]]:%[[SEL]]:%[[C1_0]], %[[C3]]:%[[C4]]:%[[C1_1]])  shape %[[SHAPE2]] : (!fir.box<!fir.array<10x?xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x2xi32>>
! CHECK:         %[[C1_2:.*]] = arith.constant 1 : index
! CHECK:         %[[C1_3:.*]] = arith.constant 1 : index
! CHECK:         %[[C10_4:.*]] = arith.constant 10 : index
! CHECK:         %[[C1_5:.*]] = arith.constant 1 : index
! CHECK:         %[[C2_6:.*]] = arith.constant 2 : index
! CHECK:         %[[C1_7:.*]] = arith.constant 1 : index
! CHECK:         %[[C2_8:.*]] = arith.constant 2 : index
! CHECK:         %[[SHAPE3:.*]] = fir.shape %[[C10_4]], %[[C2_8]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[DST:.*]] = hlfir.designate %[[DECL]]#0 (%[[C1_2]]:%[[SEL]]:%[[C1_3]], %[[C1_5]]:%[[C2_6]]:%[[C1_7]])  shape %[[SHAPE3]] : (!fir.box<!fir.array<10x?xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x2xi32>>
! CHECK:         hlfir.assign %[[SRC]] to %[[DST]] : !fir.ref<!fir.array<10x2xi32>>, !fir.ref<!fir.array<10x2xi32>>
! CHECK:         return
! CHECK:       }

subroutine assumed_size_forall_test(b)
  integer :: b(10,*)
  forall (i=2:6)
     b(i, 1:2) = b(i, 3:4)
  end forall
end subroutine assumed_size_forall_test

! CHECK-LABEL: func.func @_QPassumed_size_forall_test(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.array<10x?xi32>> {fir.bindc_name = "b"}) {
! CHECK:         %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[C10_I64:.*]] = arith.constant 10 : i64
! CHECK:         %[[IDX10:.*]] = fir.convert %[[C10_I64]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index
! CHECK:         %[[COND:.*]] = arith.cmpi sgt, %[[IDX10]], %[[C0]] : index
! CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[IDX10]], %[[C0]] : index
! CHECK:         %[[EXT:.*]] = fir.assumed_size_extent : index
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[SEL]], %[[EXT]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%[[SHAPE]]) dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFassumed_size_forall_testEb"} : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<10x?xi32>>, !fir.ref<!fir.array<10x?xi32>>)
! CHECK:         %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:         %[[C6_I32:.*]] = arith.constant 6 : i32
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %[[C2_I32]] : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %[[C6_I32]] : i32
! CHECK:         }  (%[[IARG:.*]]: i32) {
! CHECK:           %[[IDXVAR:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:           hlfir.region_assign {
! CHECK:             %[[I:.*]] = fir.load %[[IDXVAR]] : !fir.ref<i32>
! CHECK:             %[[I64:.*]] = fir.convert %[[I]] : (i32) -> i64
! CHECK:             %[[C3:.*]] = arith.constant 3 : index
! CHECK:             %[[C4:.*]] = arith.constant 4 : index
! CHECK:             %[[C1:.*]] = arith.constant 1 : index
! CHECK:             %[[C2:.*]] = arith.constant 2 : index
! CHECK:             %[[SHAPE1:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK:             %[[SRC:.*]] = hlfir.designate %[[DECL]]#0 (%[[I64]], %[[C3]]:%[[C4]]:%[[C1]])  shape %[[SHAPE1]] : (!fir.box<!fir.array<10x?xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:             hlfir.yield %[[SRC]] : !fir.box<!fir.array<2xi32>>
! CHECK:           } to {
! CHECK:             %[[I:.*]] = fir.load %[[IDXVAR]] : !fir.ref<i32>
! CHECK:             %[[I64:.*]] = fir.convert %[[I]] : (i32) -> i64
! CHECK:             %[[C1:.*]] = arith.constant 1 : index
! CHECK:             %[[C2:.*]] = arith.constant 2 : index
! CHECK:             %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:             %[[C2_1:.*]] = arith.constant 2 : index
! CHECK:             %[[SHAPE1:.*]] = fir.shape %[[C2_1]] : (index) -> !fir.shape<1>
! CHECK:             %[[DST:.*]] = hlfir.designate %[[DECL]]#0 (%[[I64]], %[[C1]]:%[[C2]]:%[[C1_0]])  shape %[[SHAPE1]] : (!fir.box<!fir.array<10x?xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:             hlfir.yield %[[DST]] : !fir.box<!fir.array<2xi32>>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }
