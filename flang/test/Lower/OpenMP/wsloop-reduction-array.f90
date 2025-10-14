! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
program reduce
integer :: i = 0
integer, dimension(2) :: r = 0

!$omp parallel do reduction(+:r)
do i=0,10
  r(1) = i
  r(2) = -i
enddo
!$omp end parallel do

print *,r
end program

! CHECK-LABEL:  omp.declare_reduction @add_reduction_byref_box_2xi32 : !fir.ref<!fir.box<!fir.array<2xi32>>> alloc {
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<2xi32>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)
! CHECK-LABEL:  } init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>):
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_1:.*]] = fir.allocmem !fir.array<2xi32> {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_3]], %[[C0]] : (!fir.box<!fir.array<2xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[SHIFT]]) : (!fir.heap<!fir.array<2xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<2xi32>>
! CHECK:           fir.store %[[VAL_7]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C1]], %[[C2]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[C1_0]] to %[[C2]] step %[[C1_0]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[SHAPE_SHIFT]]) %[[VAL_8]] : (!fir.box<!fir.array<2xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[SHAPE_SHIFT]]) %[[VAL_8]] : (!fir.box<!fir.array<2xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)
! CHECK:         }  cleanup {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<2xi32>>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<2xi32>>) -> i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:           fir.if %[[VAL_5]] {
! CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<2xi32>>) -> !fir.heap<!fir.array<2xi32>>
! CHECK:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<2xi32>>
! CHECK:           }
! CHECK:           omp.yield
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "REDUCE"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<!fir.array<2xi32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_4]]) {uniq_name = "_QFEr"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_6:.*]] = fir.embox %[[VAL_5]]#0(%[[VAL_4]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:             %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.array<2xi32>>
! CHECK:             fir.store %[[VAL_6]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:             %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_11:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_8:.*]] : !fir.ref<i32>) reduction(byref @add_reduction_byref_box_2xi32 %[[VAL_7]] -> %[[VAL_13:.*]] : !fir.ref<!fir.box<!fir.array<2xi32>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_14:.*]]) : i32 = (%[[VAL_10]]) to (%[[VAL_11]]) inclusive step (%[[VAL_12]]) {
! CHECK:                 %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.array<2xi32>>>) -> (!fir.ref<!fir.box<!fir.array<2xi32>>>, !fir.ref<!fir.box<!fir.array<2xi32>>>)
! CHECK:                 hlfir.assign %[[VAL_14]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_16:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_17:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_16]] to %[[VAL_19]] : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_20:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_21:.*]] = arith.constant 0 : i32
! CHECK:                 %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] : i32
! CHECK:                 %[[VAL_23:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_24:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_25:.*]] = hlfir.designate %[[VAL_23]] (%[[VAL_24]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_22]] to %[[VAL_25]] : i32, !fir.ref<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
