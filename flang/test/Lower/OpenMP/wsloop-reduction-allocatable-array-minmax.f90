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
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:          omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<
! CHECK:           %[[VAL_2:.*]] = arith.constant 2147483647 : i32
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
! CHECK:             %[[VAL_13:.*]] = arith.minsi %[[VAL_11]], %[[VAL_12]] : i32
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

! CHECK-LABEL:   omp.declare_reduction @max_byref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> alloc {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<
! CHECK:           %[[VAL_2:.*]] = arith.constant -2147483648 : i32
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
! CHECK:             %[[VAL_13:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_12]] : i32
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

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce15"} {
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
! CHECK:           %[[VAL_10:.*]] = arith.constant false
! CHECK:           %[[VAL_11:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_12:.*]] = fir.address_of(@_QQclXac4c37b3854f12f47cc92e78ed179316) : !fir.ref<!fir.char<1,101>>
! CHECK:           %[[VAL_13:.*]] = arith.constant 8 : i32
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_17]], %[[VAL_16]], %[[VAL_18]], %[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,101>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_22:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_20]], %[[VAL_10]], %[[VAL_11]], %[[VAL_21]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_23:.*]] = arith.constant false
! CHECK:           %[[VAL_24:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_25:.*]] = fir.address_of(@_QQclXac4c37b3854f12f47cc92e78ed179316) : !fir.ref<!fir.char<1,101>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 9 : i32
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_28:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_29:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_27]] : (index) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_30]], %[[VAL_29]], %[[VAL_31]], %[[VAL_32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,101>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_35:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_33]], %[[VAL_23]], %[[VAL_24]], %[[VAL_34]], %[[VAL_26]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_36:.*]] = arith.constant false
! CHECK:           %[[VAL_37:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[VAL_38:.*]] = fir.address_of(@_QQclXac4c37b3854f12f47cc92e78ed179316) : !fir.ref<!fir.char<1,101>>
! CHECK:           %[[VAL_39:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_41:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_42:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_43:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_41]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[VAL_43]], %[[VAL_42]], %[[VAL_44]], %[[VAL_45]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<!fir.char<1,101>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_48:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_46]], %[[VAL_36]], %[[VAL_37]], %[[VAL_47]], %[[VAL_39]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[VAL_49:.*]] = arith.constant 5 : i32
! CHECK:           hlfir.assign %[[VAL_49]] to %[[VAL_5]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_50:.*]] = arith.constant 5 : i32
! CHECK:           hlfir.assign %[[VAL_50]] to %[[VAL_7]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_51:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> index
! CHECK:           %[[VAL_53:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> index
! CHECK:           %[[VAL_55:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_56:.*]] = fir.convert %[[VAL_52]] : (index) -> i32
! CHECK:           %[[VAL_57:.*]]:2 = fir.do_loop %[[VAL_58:.*]] = %[[VAL_52]] to %[[VAL_54]] step %[[VAL_55]] iter_args(%[[VAL_59:.*]] = %[[VAL_56]]) -> (index, i32) {
! CHECK:             fir.store %[[VAL_59]] to %[[VAL_3]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_60:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_61:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:             %[[VAL_62:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i32) -> i64
! CHECK:             %[[VAL_64:.*]] = hlfir.designate %[[VAL_61]] (%[[VAL_63]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_60]] to %[[VAL_64]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_65:.*]] = arith.addi %[[VAL_58]], %[[VAL_55]] overflow<nsw> : index
! CHECK:             %[[VAL_66:.*]] = fir.convert %[[VAL_55]] : (index) -> i32
! CHECK:             %[[VAL_67:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_68:.*]] = arith.addi %[[VAL_67]], %[[VAL_66]] overflow<nsw> : i32
! CHECK:             fir.result %[[VAL_65]], %[[VAL_68]] : index, i32
! CHECK:           }
! CHECK:           fir.store %[[VAL_69:.*]]#1 to %[[VAL_3]]#1 : !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_70:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_71:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_72:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFEi_private_i32 %[[VAL_3]]#0 -> %[[VAL_73:.*]] : !fir.ref<i32>) reduction(byref @max_byref_box_heap_Uxi32 %[[VAL_5]]#0 -> %[[VAL_74:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_75:.*]]) : i32 = (%[[VAL_70]]) to (%[[VAL_71]]) inclusive step (%[[VAL_72]]) {
! CHECK:                 %[[VAL_76:.*]]:2 = hlfir.declare %[[VAL_73]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_77:.*]]:2 = hlfir.declare %[[VAL_74]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmaxes"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 hlfir.assign %[[VAL_75]] to %[[VAL_76]]#1 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_78:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_79:.*]] = arith.constant 0 : index
! CHECK:                 %[[VAL_80:.*]]:3 = fir.box_dims %[[VAL_78]], %[[VAL_79]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                 %[[VAL_81:.*]] = fir.shape %[[VAL_80]]#1 : (index) -> !fir.shape<1>
! CHECK:                 %[[VAL_82:.*]] = fir.load %[[VAL_77]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_83:.*]] = hlfir.elemental %[[VAL_81]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:                 ^bb0(%[[VAL_84:.*]]: index):
! CHECK:                   %[[VAL_85:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_86:.*]]:3 = fir.box_dims %[[VAL_78]], %[[VAL_85]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_87:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_88:.*]] = arith.subi %[[VAL_86]]#0, %[[VAL_87]] : index
! CHECK:                   %[[VAL_89:.*]] = arith.addi %[[VAL_84]], %[[VAL_88]] : index
! CHECK:                   %[[VAL_90:.*]] = hlfir.designate %[[VAL_78]] (%[[VAL_89]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                   %[[VAL_91:.*]] = fir.load %[[VAL_90]] : !fir.ref<i32>
! CHECK:                   %[[VAL_92:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_93:.*]]:3 = fir.box_dims %[[VAL_82]], %[[VAL_92]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_94:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_95:.*]] = arith.subi %[[VAL_93]]#0, %[[VAL_94]] : index
! CHECK:                   %[[VAL_96:.*]] = arith.addi %[[VAL_84]], %[[VAL_95]] : index
! CHECK:                   %[[VAL_97:.*]] = hlfir.designate %[[VAL_82]] (%[[VAL_96]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                   %[[VAL_98:.*]] = fir.load %[[VAL_97]] : !fir.ref<i32>
! CHECK:                   %[[VAL_99:.*]] = arith.cmpi sgt, %[[VAL_91]], %[[VAL_98]] : i32
! CHECK:                   %[[VAL_100:.*]] = arith.select %[[VAL_99]], %[[VAL_91]], %[[VAL_98]] : i32
! CHECK:                   hlfir.yield_element %[[VAL_100]] : i32
! CHECK:                 }
! CHECK:                 hlfir.assign %[[VAL_83]] to %[[VAL_77]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 hlfir.destroy %[[VAL_83]] : !hlfir.expr<?xi32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_101:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_102:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_103:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@_QFEi_private_i32 %[[VAL_3]]#0 -> %[[VAL_104:.*]] : !fir.ref<i32>) reduction(byref @min_byref_box_heap_Uxi32 %[[VAL_7]]#0 -> %[[VAL_105:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:               omp.loop_nest (%[[VAL_106:.*]]) : i32 = (%[[VAL_101]]) to (%[[VAL_102]]) inclusive step (%[[VAL_103]]) {
! CHECK:                 %[[VAL_107:.*]]:2 = hlfir.declare %[[VAL_104]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_108:.*]]:2 = hlfir.declare %[[VAL_105]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEmins"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 hlfir.assign %[[VAL_106]] to %[[VAL_107]]#1 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_109:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_110:.*]] = arith.constant 0 : index
! CHECK:                 %[[VAL_111:.*]]:3 = fir.box_dims %[[VAL_109]], %[[VAL_110]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                 %[[VAL_112:.*]] = fir.shape %[[VAL_111]]#1 : (index) -> !fir.shape<1>
! CHECK:                 %[[VAL_113:.*]] = fir.load %[[VAL_108]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_114:.*]] = hlfir.elemental %[[VAL_112]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:                 ^bb0(%[[VAL_115:.*]]: index):
! CHECK:                   %[[VAL_116:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_117:.*]]:3 = fir.box_dims %[[VAL_109]], %[[VAL_116]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_118:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_119:.*]] = arith.subi %[[VAL_117]]#0, %[[VAL_118]] : index
! CHECK:                   %[[VAL_120:.*]] = arith.addi %[[VAL_115]], %[[VAL_119]] : index
! CHECK:                   %[[VAL_121:.*]] = hlfir.designate %[[VAL_109]] (%[[VAL_120]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                   %[[VAL_122:.*]] = fir.load %[[VAL_121]] : !fir.ref<i32>
! CHECK:                   %[[VAL_123:.*]] = arith.constant 0 : index
! CHECK:                   %[[VAL_124:.*]]:3 = fir.box_dims %[[VAL_113]], %[[VAL_123]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:                   %[[VAL_125:.*]] = arith.constant 1 : index
! CHECK:                   %[[VAL_126:.*]] = arith.subi %[[VAL_124]]#0, %[[VAL_125]] : index
! CHECK:                   %[[VAL_127:.*]] = arith.addi %[[VAL_115]], %[[VAL_126]] : index
! CHECK:                   %[[VAL_128:.*]] = hlfir.designate %[[VAL_113]] (%[[VAL_127]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                   %[[VAL_129:.*]] = fir.load %[[VAL_128]] : !fir.ref<i32>
! CHECK:                   %[[VAL_130:.*]] = arith.cmpi slt, %[[VAL_122]], %[[VAL_129]] : i32
! CHECK:                   %[[VAL_131:.*]] = arith.select %[[VAL_130]], %[[VAL_122]], %[[VAL_129]] : i32
! CHECK:                   hlfir.yield_element %[[VAL_131]] : i32
! CHECK:                 }
! CHECK:                 hlfir.assign %[[VAL_114]] to %[[VAL_108]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 hlfir.destroy %[[VAL_114]] : !hlfir.expr<?xi32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           %[[VAL_132:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_133:.*]] = fir.address_of(@_QQclXac4c37b3854f12f47cc92e78ed179316) : !fir.ref<!fir.char<1,101>>
! CHECK:           %[[VAL_134:.*]] = fir.convert %[[VAL_133]] : (!fir.ref<!fir.char<1,101>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_135:.*]] = arith.constant 31 : i32
! CHECK:           %[[VAL_136:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_132]], %[[VAL_134]], %[[VAL_135]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_137:.*]] = fir.address_of(@_QQclX6D61783A20) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_138:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_139:.*]]:2 = hlfir.declare %[[VAL_137]] typeparams %[[VAL_138]] {fortran_attrs = {{.*}}<parameter>, uniq_name = "_QQclX6D61783A20"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_140:.*]] = fir.convert %[[VAL_139]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_141:.*]] = fir.convert %[[VAL_138]] : (index) -> i64
! CHECK:           %[[VAL_142:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_136]], %[[VAL_140]], %[[VAL_141]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_143:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_144:.*]] = fir.convert %[[VAL_143]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_145:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_136]], %[[VAL_144]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_146:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_136]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           %[[VAL_147:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_148:.*]] = fir.address_of(@_QQclXac4c37b3854f12f47cc92e78ed179316) : !fir.ref<!fir.char<1,101>>
! CHECK:           %[[VAL_149:.*]] = fir.convert %[[VAL_148]] : (!fir.ref<!fir.char<1,101>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_150:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_151:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_147]], %[[VAL_149]], %[[VAL_150]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[VAL_152:.*]] = fir.address_of(@_QQclX6D696E3A20) : !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_153:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_154:.*]]:2 = hlfir.declare %[[VAL_152]] typeparams %[[VAL_153]] {fortran_attrs = {{.*}}<parameter>, uniq_name = "_QQclX6D696E3A20"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_155:.*]] = fir.convert %[[VAL_154]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_156:.*]] = fir.convert %[[VAL_153]] : (index) -> i64
! CHECK:           %[[VAL_157:.*]] = fir.call @_FortranAioOutputAscii(%[[VAL_151]], %[[VAL_155]], %[[VAL_156]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
! CHECK:           %[[VAL_158:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_159:.*]] = fir.convert %[[VAL_158]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<none>
! CHECK:           %[[VAL_160:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_151]], %[[VAL_159]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[VAL_161:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_151]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }
