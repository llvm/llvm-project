! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
program reduce
integer :: i = 0
integer, dimension(2) :: r = 0

!$omp parallel do reduction(+:r)
do i=0,10
  r(1) = r(1) + i
  r(2) = r(2) - i
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
! CHECK:           %[[TRUE:.*]]  = arith.constant true
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_3]], %[[C0]] : (!fir.box<!fir.array<2xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[SHIFT]]) : (!fir.heap<!fir.array<2xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<2xi32>>
! CHECK:           fir.store %[[VAL_7]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)

! CHECK:        } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<2xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_5]]#1 step %[[VAL_7]] unordered {
! CHECK:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<2xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_6]]) %[[VAL_8]] : (!fir.box<!fir.array<2xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
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

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
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
! CHECK:             %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK:             %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_11:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(byref @add_reduction_byref_box_2xi32 %[[VAL_7]] -> %[[VAL_13:.*]] : !fir.ref<!fir.box<!fir.array<2xi32>>>) {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_14:.*]]) : i32 = (%[[VAL_10]]) to (%[[VAL_11]]) inclusive step (%[[VAL_12]]) {
! CHECK:                 %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.array<2xi32>>>) -> (!fir.ref<!fir.box<!fir.array<2xi32>>>, !fir.ref<!fir.box<!fir.array<2xi32>>>)
! CHECK:                 fir.store %[[VAL_14]] to %[[VAL_9]]#1 : !fir.ref<i32>
! CHECK:                 %[[VAL_16:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_18:.*]] = hlfir.designate %[[VAL_16]] (%[[VAL_17]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
! CHECK:                 %[[VAL_20:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_21:.*]] = arith.addi %[[VAL_19]], %[[VAL_20]] : i32
! CHECK:                 %[[VAL_22:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:                 %[[VAL_24:.*]] = hlfir.designate %[[VAL_22]] (%[[VAL_23]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_21]] to %[[VAL_24]] : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_25:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_26:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_27:.*]] = hlfir.designate %[[VAL_25]] (%[[VAL_26]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<i32>
! CHECK:                 %[[VAL_29:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_30:.*]] = arith.subi %[[VAL_28]], %[[VAL_29]] : i32
! CHECK:                 %[[VAL_31:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:                 %[[VAL_32:.*]] = arith.constant 2 : index
! CHECK:                 %[[VAL_33:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:                 hlfir.assign %[[VAL_30]] to %[[VAL_33]] : i32, !fir.ref<i32>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
