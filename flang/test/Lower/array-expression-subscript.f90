! RUN: bbc --emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1a(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_10]] : index
! CHECK:         %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_12]] : index
! CHECK:         %[[VAL_18:.*]] = arith.divsi %[[VAL_17]], %[[VAL_12]] : index
! CHECK:         %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_15]] : index
! CHECK:         %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_18]], %[[VAL_15]] : index
! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_22:.*]] = fir.slice %[[VAL_10]], %[[VAL_14]], %[[VAL_12]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_2]](%[[VAL_21]]) {{\[}}%[[VAL_22]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_3]] : index
! CHECK:         %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_3]], %[[VAL_20]] : index
! CHECK:         %[[VAL_32:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_34:.*]] = fir.array_load %[[VAL_1]](%[[VAL_32]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_35:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_37:.*]] = arith.subi %[[VAL_25]], %[[VAL_35]] : index
! CHECK:         %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[VAL_35]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_7]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_41:.*]] = fir.array_fetch %[[VAL_23]], %[[VAL_39]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
! CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_42]], %[[VAL_8]] : index
! CHECK:           %[[VAL_44:.*]] = fir.array_fetch %[[VAL_34]], %[[VAL_43]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_45:.*]] = fir.array_update %[[VAL_40]], %[[VAL_44]], %[[VAL_39]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_45]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_7]], %[[VAL_46:.*]] to %[[VAL_0]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }

subroutine test1a(a,b,c)
  integer :: a(10), b(10), c(20)

  a = b(c(1:20:2))
end subroutine test1a

! CHECK-LABEL: func @_QPtest1b(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_8]] : index
! CHECK:         %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_10]] : index
! CHECK:         %[[VAL_16:.*]] = arith.divsi %[[VAL_15]], %[[VAL_10]] : index
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_13]] : index
! CHECK:         %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_13]] : index
! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_20:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_2]](%[[VAL_19]]) {{\[}}%[[VAL_20]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_4]] : index
! CHECK:         %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_4]], %[[VAL_18]] : index
! CHECK:         %[[VAL_30:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_32:.*]] = fir.array_load %[[VAL_1]](%[[VAL_30]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_33:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_34:.*]] = fir.array_load %[[VAL_0]](%[[VAL_33]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_35:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_37:.*]] = arith.subi %[[VAL_23]], %[[VAL_35]] : index
! CHECK:         %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[VAL_35]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_32]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_41:.*]] = fir.array_fetch %[[VAL_34]], %[[VAL_39]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_42:.*]] = fir.array_fetch %[[VAL_21]], %[[VAL_39]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> index
! CHECK:           %[[VAL_44:.*]] = arith.subi %[[VAL_43]], %[[VAL_6]] : index
! CHECK:           %[[VAL_45:.*]] = fir.array_update %[[VAL_40]], %[[VAL_41]], %[[VAL_44]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_45]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_32]], %[[VAL_46:.*]] to %[[VAL_1]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }

subroutine test1b(a,b,c)
  integer :: a(10), b(10), c(20)

  b(c(1:20:2)) = a
end subroutine test1b

! CHECK-LABEL: func @_QPtest2a(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]](%[[VAL_8]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_3]](%[[VAL_12]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_6]] : index
! CHECK:         %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_2]](%[[VAL_16]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_4]] : index
! CHECK:         %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_4]], %[[VAL_15]] : index
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_29:.*]] = fir.array_load %[[VAL_1]](%[[VAL_27]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_32:.*]] = arith.subi %[[VAL_20]], %[[VAL_30]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_9]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_34]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_11]] : index
! CHECK:           %[[VAL_39:.*]] = fir.array_fetch %[[VAL_18]], %[[VAL_38]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> index
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_40]], %[[VAL_10]] : index
! CHECK:           %[[VAL_42:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_41]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_43:.*]] = fir.array_update %[[VAL_35]], %[[VAL_42]], %[[VAL_34]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_43]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_44:.*]] to %[[VAL_0]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }

subroutine test2a(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  a = b(c(d))
end subroutine test2a

! CHECK-LABEL: func @_QPtest2b(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_3]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_6]] : index
! CHECK:         %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_2]](%[[VAL_14]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_5]] : index
! CHECK:         %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_5]], %[[VAL_13]] : index
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_27:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_28:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_29:.*]] = fir.array_load %[[VAL_0]](%[[VAL_28]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_32:.*]] = arith.subi %[[VAL_18]], %[[VAL_30]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_27]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_34]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_37:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_34]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> index
! CHECK:           %[[VAL_39:.*]] = arith.subi %[[VAL_38]], %[[VAL_9]] : index
! CHECK:           %[[VAL_40:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_39]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> index
! CHECK:           %[[VAL_42:.*]] = arith.subi %[[VAL_41]], %[[VAL_8]] : index
! CHECK:           %[[VAL_43:.*]] = fir.array_update %[[VAL_35]], %[[VAL_36]], %[[VAL_42]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_43]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_27]], %[[VAL_44:.*]] to %[[VAL_1]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }

subroutine test2b(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  b(c(d)) = a
end subroutine test2b

! CHECK-LABEL: func @_QPtest1c(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>{{.*}}, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
! CHECK:         return
! CHECK:       }
subroutine test1c(a,b,c,d)
  integer :: a(10), b(10), d(10), c(20)

  ! flang: parser FAIL (final position)
  !a = b(d(c(1:20:2))
end subroutine test1c
