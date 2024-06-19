! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test that the constant array expression actual argument
! is placed into a temporary inside 'test' subroutine before
! the call to 'sub'. This is required because the compiler-generated
! copy-out inside 'sub' after the call to 'sub2' will write
! into the dummy argument 'i'. If the constant array expression is passed
! directly byref to 'sub', the copy-out will try to modify readonly memory.
subroutine sub(i, n)
  interface
     subroutine sub2(i)
       integer :: i(*)
     end subroutine sub2
  end interface
  integer :: i(n)
  call sub2(i(3::2))
end subroutine sub
! CHECK-LABEL:   func.func @_QPsub(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "i"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFsubEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_9]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFsubEi"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_12:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_8]], %[[VAL_11]] : index
! CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_12]] : index
! CHECK:           %[[VAL_16:.*]] = arith.divsi %[[VAL_15]], %[[VAL_12]] : index
! CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_13]] : index
! CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_13]] : index
! CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_11]]:%[[VAL_8]]:%[[VAL_12]])  shape %[[VAL_19]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.copy_in %[[VAL_20]] to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.box<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_22:.*]] = fir.box_addr %[[VAL_21]]#0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           fir.call @_QPsub2(%[[VAL_22]]) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_21]]#1 to %[[VAL_20]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i1, !fir.box<!fir.array<?xi32>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test
  call sub((/1,2,3,4,5/), 5)
end subroutine test
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QQro.5xi4.0) : !fir.ref<!fir.array<5xi32>>
! CHECK:           %[[VAL_1:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.5xi4.0"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_5:.*]] = hlfir.as_expr %[[VAL_3]]#0 : (!fir.ref<!fir.array<5xi32>>) -> !hlfir.expr<5xi32>
! CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_2]]) {adapt.valuebyref} : (!hlfir.expr<5xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>, i1)
! CHECK:           %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_4]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.array<5xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           fir.call @_QPsub(%[[VAL_8]], %[[VAL_7]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<!fir.array<5xi32>>, i1
! CHECK:           hlfir.end_associate %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<i32>, i1
! CHECK:           return
! CHECK:         }
