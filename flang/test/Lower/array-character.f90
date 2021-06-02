! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPissue
subroutine issue(c1, c2)
  ! CHECK-DAG: %[[VAL_0:.*]] = constant 3 : index
  ! CHECK-DAG: %[[VAL_1:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_2:.*]] = constant 1 : index
  ! CHECK: %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_4:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK: %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_7:.*]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
  ! CHECK: %[[VAL_9:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
  ! CHECK: br ^bb1(%[[VAL_1]], %[[VAL_0]] : index, index)
  ! CHECK: ^bb1(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
  ! CHECK: %[[VAL_12:.*]] = cmpi sgt, %[[VAL_11]], %[[VAL_1]] : index
  ! CHECK: cond_br %[[VAL_12]], ^bb2, ^bb3
  ! CHECK: ^bb2
  ! CHECK: %[[VAL_13:.*]] = addi %[[VAL_10]], %[[VAL_2]] : index
  ! CHECK: %[[VAL_14:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_9]]) %[[VAL_13]] typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,4>>
  ! CHECK: %[[VAL_16:.*]] = fir.array_coor %[[VAL_5]](%[[VAL_9]]) %[[VAL_13]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,4>>
  ! CHECK: %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.char<1,4>>
  ! CHECK: fir.store %[[VAL_17]] to %[[VAL_16]] : !fir.ref<!fir.char<1,4>>
  ! CHECK: %[[VAL_18:.*]] = subi %[[VAL_11]], %[[VAL_2]] : index
  ! CHECK: br ^bb1(%[[VAL_13]], %[[VAL_18]] : index, index)
  ! CHECK: ^bb3
  ! CHECK: return
  character(4) :: c1(3)
  character(*) :: c2(3)
  c1 = c2
end subroutine

program p
  character(4) :: c1(3)
  character(4) :: c2(3) = ["abcd", "    ", "    "]
  print *, c2
  call issue(c1, c2)
  print *, c1
end program p

