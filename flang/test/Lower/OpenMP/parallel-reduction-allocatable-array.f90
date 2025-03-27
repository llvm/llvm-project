! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program reduce
integer :: i = 0
integer, dimension(:), allocatable :: r

allocate(r(2))

!$omp parallel do reduction(+:r)
do i=0,10
  r(1) = i
  r(2) = -i
enddo
!$omp end parallel do

print *,r

end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> alloc {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_10:.*]] = fir.embox %[[VAL_4]](%[[VAL_9]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_13:.*]] = fir.shape %[[VAL_12]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_14:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_12]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:             %[[VAL_15:.*]] = arith.constant true
! CHECK:             %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_14]](%[[VAL_13]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_17]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_19:.*]] = fir.shape_shift %[[VAL_18]]#0, %[[VAL_18]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_20:.*]] = fir.rebox %[[VAL_16]]#0(%[[VAL_19]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_2]] to %[[VAL_20]] : i32, !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_20]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[C1]], %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             fir.freemem %[[VAL_2]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant false
! CHECK:           %[[VAL_5:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QQclX631fe30e1f6e089e15ae2bda4e306e13) : !fir.ref<!fir.char<1,96>>
! CHECK:           %[[VAL_7:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_8]] : (index) -> i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_11]], %[[VAL_10]], %[[VAL_12]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_16:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_14]], %[[VAL_4]], %[[VAL_5]], %[[VAL_15]], %[[VAL_7]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_18:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFEi_private_i32 %[[VAL_1]]#0 -> %[[VAL_20:.*]] : !fir.ref<i32>) reduction(byref @add_reduction_byref_box_heap_Uxi32 %[[VAL_3]]#0 -> %[[VAL_21:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_22:.*]]) : i32 = (%[[VAL_17]]) to (%[[VAL_18]]) inclusive step (%[[VAL_19]]) {
! CHECK:                 %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_20]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_21]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 hlfir.assign %[[VAL_22]] to %[[VAL_23]]#1 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_25:.*]] = fir.load %[[VAL_23]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_26:.*]] = fir.load %[[VAL_24]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_28:.*]] = hlfir.designate %[[VAL_26]] (%[[VAL_27]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_25]] to %[[VAL_28]] : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_29:.*]] = fir.load %[[VAL_23]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_30:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_29]] : i32
! CHECK:                 %[[VAL_32:.*]] = fir.load %[[VAL_24]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_33:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_34:.*]] = hlfir.designate %[[VAL_32]] (%[[VAL_33]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_31]] to %[[VAL_34]] : i32, !fir.ref<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[VAL_35:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_36:.*]] = fir.address_of(@_QQclX631fe30e1f6e089e15ae2bda4e306e13) : !fir.ref<!fir.char<1,96>>
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_38:.*]] = arith.constant 17 : i32
! CHECK:           %[[VAL_39:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_35]], %[[VAL_37]], %[[VAL_38]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_42:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_39]], %[[VAL_41]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_43:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_39]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }
