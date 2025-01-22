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
! CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           omp.yield(%[[VAL_10]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK-LABEL:   } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[ADDR:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[ADDRI:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:           %[[IS_NULL:.*]] = arith.cmpi eq, %[[ADDRI]], %[[C0_I64]] : i64
! CHECK:           fir.if %[[IS_NULL]] {
! CHECK:             %[[NULL_BOX:.*]] = fir.embox %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[NULL_BOX]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           } else {
! CHECK:             %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_6:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_4]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:             %[[VAL_7:.*]] = arith.constant true
! CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:             %[[C0:.*]] = arith.constant 0 : index
! CHECK:             %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_2]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:             %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:             %[[REBOX:.*]] = fir.rebox %[[VAL_8]]#0(%[[SHIFT]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             hlfir.assign %[[VAL_1]] to %[[REBOX]] : i32, !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:             fir.store %[[REBOX]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           }
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
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
! CHECK:         }  cleanup {
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

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : index
! CHECK:           %[[VAL_9:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_8]] {fir.must_be_heap = true, uniq_name = "_QFEr.alloc"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_15:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_heap_Uxi32 %[[VAL_3]]#0 -> %[[VAL_17:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_18:.*]]) : i32 = (%[[VAL_14]]) to (%[[VAL_15]]) inclusive step (%[[VAL_16]]) {
! CHECK:                 %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_17]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:                 fir.store %[[VAL_18]] to %[[VAL_13]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_20:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_21:.*]] = fir.load %[[VAL_19]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_23:.*]] = hlfir.designate %[[VAL_21]] (%[[VAL_22]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_20]] to %[[VAL_23]] : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_24:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_25:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_26:.*]] = arith.subi %[[VAL_25]], %[[VAL_24]] : i32
! CHECK:                 %[[VAL_27:.*]] = fir.load %[[VAL_19]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:                 %[[VAL_28:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_29:.*]] = hlfir.designate %[[VAL_27]] (%[[VAL_28]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_26]] to %[[VAL_29]] : i32, !fir.ref<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
