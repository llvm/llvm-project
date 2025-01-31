! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

program reduce
integer :: i = 0
integer, dimension(:), pointer :: r

allocate(r(2))

!$omp parallel do reduction(+:r)
do i=0,10
  r(1) = i
  r(2) = -i
enddo
!$omp end parallel do

print *,r
deallocate(r)

end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_ptr_Uxi32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> alloc {
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[C0:.*]] = arith.constant 0 : index
! CHECK:             %[[SHAPE:.*]] = fir.shape %[[C0]]
! CHECK:             %[[VAL_8:.*]] = fir.embox %[[VAL_4]](%[[SHAPE]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_8]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_9]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_11:.*]] = fir.shape %[[VAL_10]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_12:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:             %[[VAL_13:.*]] = arith.constant true
! CHECK:             %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_11]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_15]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_17:.*]] = fir.shape_shift %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_18:.*]] = fir.rebox %[[VAL_14]]#0(%[[VAL_17]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_1]] to %[[VAL_18]] : i32, !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_18]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           }
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant false
! CHECK:           %[[VAL_5:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_6:.*]] = fir.address_of(
! CHECK:           %[[VAL_7:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_8]](%[[VAL_10]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:           fir.call @_FortranAPointerSetBounds(%[[VAL_15]], %[[VAL_14]], %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_21:.*]] = fir.call @_FortranAPointerAllocate(%[[VAL_19]], %[[VAL_4]], %[[VAL_5]], %[[VAL_20]], %[[VAL_7]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_22:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
! CHECK:             %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_22]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_24:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_25:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_26:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_ptr_Uxi32 %[[VAL_3]]#0 -> %[[VAL_27:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_28:.*]]) : i32 = (%[[VAL_24]]) to (%[[VAL_25]]) inclusive step (%[[VAL_26]]) {
! CHECK:                 %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_27]] {fortran_attrs = {{.*}}<pointer>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:                 fir.store %[[VAL_28]] to %[[VAL_23]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_30:.*]] = fir.load %[[VAL_23]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_31:.*]] = fir.load %[[VAL_29]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_32:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_33:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_30]] to %[[VAL_33]] : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_34:.*]] = fir.load %[[VAL_23]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_35:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_34]] : i32
! CHECK:                 %[[VAL_37:.*]] = fir.load %[[VAL_29]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_38:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_39:.*]] = hlfir.designate %[[VAL_37]] (%[[VAL_38]])  : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_36]] to %[[VAL_39]] : i32, !fir.ref<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
