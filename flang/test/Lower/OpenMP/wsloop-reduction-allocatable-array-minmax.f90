! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
program reduce15
  integer, parameter :: SIZE = 10
  integer, dimension(:), allocatable :: arr,maxes,mins
  integer :: i

  allocate(arr(10))
  allocate(maxes(10))
  allocate(mins(10))

  maxes = 5
  mins = 5
  do i = 1,SIZE
    arr(i) = i
  end do

  !$omp parallel do reduction(max:maxes)
  do i = 1,SIZE
    maxes = max(arr, maxes)
  end do
  !$omp end parallel do


  !$omp parallel do reduction(min:mins)
  do i = 1,SIZE
    mins = min(arr, mins)
  end do
  !$omp end parallel do

  print *,"max: ", maxes
  print *,"min: ", mins
end program

! CHECK-LABEL:   omp.declare_reduction @min_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> alloc {
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 2147483647 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[C0:.*]] = arith.constant 0 : index
! CHECK:             %[[SHAPE:.*]] = fir.shape %[[C0]]
! CHECK:             %[[VAL_8:.*]] = fir.embox %[[VAL_4]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_8]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_11:.*]] = fir.shape %[[VAL_10]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_12:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:             %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_11]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_17:.*]] = fir.shape_shift %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_18:.*]] = fir.rebox %[[VAL_14]]#0(%[[VAL_17]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_1]] to %[[VAL_18]] : i32, !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_18]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           }
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
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
! CHECK:             %[[VAL_13:.*]] = arith.minsi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
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

! CHECK-LABEL:   omp.declare_reduction @max_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> alloc {
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant -2147483648 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           fir.if %[[VAL_7]] {
! CHECK:             %[[C0:.*]] = arith.constant 0 : index
! CHECK:             %[[SHAPE:.*]] = fir.shape %[[C0]]
! CHECK:             %[[VAL_8:.*]] = fir.embox %[[VAL_4]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_8]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_11:.*]] = fir.shape %[[VAL_10]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_12:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:             %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_11]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_17:.*]] = fir.shape_shift %[[VAL_16]]#0, %[[VAL_16]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[VAL_18:.*]] = fir.rebox %[[VAL_14]]#0(%[[VAL_17]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_1]] to %[[VAL_18]] : i32, !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[VAL_18]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           }
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
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
! CHECK:             %[[VAL_13:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
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

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "REDUCE15"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QFEmaxes) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmaxes"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QFEmins) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmins"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFECsize) : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {fortran_attrs = {{.*}}<parameter>, uniq_name = "_QFECsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:           %[[VAL_15:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_14]] {fir.must_be_heap = true, uniq_name = "_QFEarr.alloc"}
! CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_17]] to %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_18:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_20]] : index
! CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_19]], %[[VAL_20]] : index
! CHECK:           %[[VAL_23:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_22]] {fir.must_be_heap = true, uniq_name = "_QFEmaxes.alloc"}
! CHECK:           %[[VAL_24:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_25:.*]] = fir.embox %[[VAL_23]](%[[VAL_24]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_25]] to %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> index
! CHECK:           %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_28]] : index
! CHECK:           %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_27]], %[[VAL_28]] : index
! CHECK:           %[[VAL_31:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_30]] {fir.must_be_heap = true, uniq_name = "_QFEmins.alloc"}
! CHECK:           %[[VAL_32:.*]] = fir.shape %[[VAL_30]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_33:.*]] = fir.embox %[[VAL_31]](%[[VAL_32]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_33]] to %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_34:.*]] = arith.constant 5 : i32
! CHECK:           hlfir.assign %[[VAL_34]] to %[[VAL_5]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_35:.*]] = arith.constant 5 : i32
! CHECK:           hlfir.assign %[[VAL_35]] to %[[VAL_7]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:           %[[VAL_38:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> index
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_37]] : (index) -> i32
! CHECK:           %[[VAL_42:.*]] = fir.do_loop %[[VAL_43:.*]] = %[[VAL_37]] to %[[VAL_39]] step %[[VAL_40]] iter_args(%[[VAL_44:.*]] = %[[VAL_41]]) -> (i32) {
! CHECK:             fir.store %[[VAL_44]] to %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_45:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_46:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
! CHECK:             %[[VAL_49:.*]] = hlfir.designate %[[VAL_46]] (%[[VAL_48]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_45]] to %[[VAL_49]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_40]] : (index) -> i32
! CHECK:             %[[VAL_52:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_51]] overflow<nsw> : i32
! CHECK:             fir.result %[[VAL_53]] : i32
! CHECK:           }
! CHECK:           fir.store %[[VAL_54:.*]] to %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_57:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_58:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_59:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_55:.*]] : !fir.ref<i32>) reduction(byref @max_byref_box_heap_Uxi32 %[[VAL_5]]#0 -> %[[VAL_60:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_61:.*]]) : i32 = (%[[VAL_57]]) to (%[[VAL_58]]) inclusive step (%[[VAL_59]]) {
! CHECK:                 %[[VAL_56:.*]]:2 = hlfir.declare %[[VAL_55]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_62:.*]]:2 = hlfir.declare %[[VAL_60]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmaxes"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 hlfir.assign %[[VAL_61]] to %[[VAL_56]]#0 : i32, !fir.ref<i32>
! CHECK-DAG:             %[[VAL_63:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:             %[[VAL_64:.*]] = arith.constant 0 : index
! CHECK-DAG:             %[[VAL_65:.*]]:3 = fir.box_dims %[[VAL_63]], %[[VAL_64]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK-DAG:             %[[VAL_66:.*]] = fir.shape %[[VAL_65]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:             %[[VAL_67:.*]] = fir.load %[[VAL_62]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_68:.*]] = hlfir.elemental %[[VAL_66]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:                 ^bb0(%[[VAL_69:.*]]: index):
! CHECK:                   %[[VAL_70:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_71:.*]]:3 = fir.box_dims %[[VAL_63]], %[[VAL_70]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_72:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_73:.*]] = arith.subi %[[VAL_71]]#0, %[[VAL_72]] : index
! CHECK:                   %[[VAL_74:.*]] = arith.addi %[[VAL_69]], %[[VAL_73]] : index
! CHECK-DAG:                   %[[VAL_75:.*]] = hlfir.designate %[[VAL_63]] (%[[VAL_74]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK-DAG:                   %[[VAL_76:.*]] = fir.load %[[VAL_75]] : !fir.ref<i32>
! CHECK-DAG:                   %[[VAL_77:.*]] = arith.constant 0 : index
! CHECK-DAG:                   %[[VAL_78:.*]]:3 = fir.box_dims %[[VAL_67]], %[[VAL_77]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK-DAG:                   %[[VAL_79:.*]] = arith.constant 1 : index
! CHECK-DAG:                   %[[VAL_80:.*]] = arith.subi %[[VAL_78]]#0, %[[VAL_79]] : index
! CHECK-DAG:                   %[[VAL_81:.*]] = arith.addi %[[VAL_69]], %[[VAL_80]] : index
! CHECK-DAG:                   %[[VAL_82:.*]] = hlfir.designate %[[VAL_67]] (%[[VAL_81]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK-DAG:                   %[[VAL_83:.*]] = fir.load %[[VAL_82]] : !fir.ref<i32>
! CHECK:                   %[[VAL_84:.*]] = arith.cmpi sgt, %[[VAL_76]], %[[VAL_83]] : i32
! CHECK:                   %[[VAL_85:.*]] = arith.select %[[VAL_84]], %[[VAL_76]], %[[VAL_83]] : i32
! CHECK:                   hlfir.yield_element %[[VAL_85]] : i32
! CHECK:                 }
! CHECK:                 hlfir.assign %[[VAL_68]] to %[[VAL_62]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 hlfir.destroy %[[VAL_68]] : !hlfir.expr<?xi32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_89:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_90:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_91:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_87:.*]] : !fir.ref<i32>) reduction(byref @min_byref_box_heap_Uxi32 %[[VAL_7]]#0 -> %[[VAL_92:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_93:.*]]) : i32 = (%[[VAL_89]]) to (%[[VAL_90]]) inclusive step (%[[VAL_91]]) {
! CHECK:                 %[[VAL_88:.*]]:2 = hlfir.declare %[[VAL_87]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_94:.*]]:2 = hlfir.declare %[[VAL_92]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmins"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 hlfir.assign %[[VAL_93]] to %[[VAL_88]]#0 : i32, !fir.ref<i32>
! CHECK-DAG:                 %[[VAL_95:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:                 %[[VAL_96:.*]] = arith.constant 0 : index
! CHECK-DAG:                 %[[VAL_97:.*]]:3 = fir.box_dims %[[VAL_95]], %[[VAL_96]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK-DAG:                 %[[VAL_98:.*]] = fir.shape %[[VAL_97]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:                 %[[VAL_99:.*]] = fir.load %[[VAL_94]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:                 %[[VAL_100:.*]] = hlfir.elemental %[[VAL_98]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:                 ^bb0(%[[VAL_101:.*]]: index):
! CHECK:                   %[[VAL_102:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_103:.*]]:3 = fir.box_dims %[[VAL_95]], %[[VAL_102]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_104:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_105:.*]] = arith.subi %[[VAL_103]]#0, %[[VAL_104]] : index
! CHECK:                   %[[VAL_106:.*]] = arith.addi %[[VAL_101]], %[[VAL_105]] : index
! CHECK-DAG:               %[[VAL_107:.*]] = hlfir.designate %[[VAL_95]] (%[[VAL_106]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK-DAG:               %[[VAL_108:.*]] = fir.load %[[VAL_107]] : !fir.ref<i32>
! CHECK-DAG:               %[[VAL_109:.*]] = arith.constant 0 : index
! CHECK-DAG:               %[[VAL_110:.*]]:3 = fir.box_dims %[[VAL_99]], %[[VAL_109]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK-DAG:               %[[VAL_111:.*]] = arith.constant 1 : index
! CHECK-DAG:               %[[VAL_112:.*]] = arith.subi %[[VAL_110]]#0, %[[VAL_111]] : index
! CHECK-DAG:               %[[VAL_113:.*]] = arith.addi %[[VAL_101]], %[[VAL_112]] : index
! CHECK-DAG:               %[[VAL_114:.*]] = hlfir.designate %[[VAL_99]] (%[[VAL_113]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK-DAG:               %[[VAL_115:.*]] = fir.load %[[VAL_114]] : !fir.ref<i32>
! CHECK:                   %[[VAL_116:.*]] = arith.cmpi slt, %[[VAL_108]], %[[VAL_115]] : i32
! CHECK:                   %[[VAL_117:.*]] = arith.select %[[VAL_116]], %[[VAL_108]], %[[VAL_115]] : i32
! CHECK:                   hlfir.yield_element %[[VAL_117]] : i32
! CHECK:                 }
! CHECK:                 hlfir.assign %[[VAL_100]] to %[[VAL_94]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 hlfir.destroy %[[VAL_100]] : !hlfir.expr<?xi32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
