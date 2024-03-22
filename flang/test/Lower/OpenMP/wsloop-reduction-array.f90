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

! CHECK-LABEL   omp.declare_reduction @add_reduction_byref_box_2xi32 : !fir.ref<!fir.box<!fir.array<2xi32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<2xi32>>>):
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<2xi32> {bindc_name = ".tmp"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[VAL_5]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<2xi32>>
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<2xi32>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)

! CHECK-LABEL   } combiner {
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
! CHECK:         }

! CHECK-LABEL   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<!fir.array<2xi32>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_4]]) {uniq_name = "_QFEr"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_6:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_11:.*]] = fir.embox %[[VAL_5]]#1(%[[VAL_4]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:             %[[VAL_12:.*]] = fir.alloca !fir.box<!fir.array<2xi32>>
! CHECK:             fir.store %[[VAL_11]] to %[[VAL_12]] : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:             omp.wsloop byref reduction(@add_reduction_byref_box_2xi32 %[[VAL_12]] -> %[[VAL_13:.*]] : !fir.ref<!fir.box<!fir.array<2xi32>>>)  for  (%[[VAL_14:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_7]]#1 : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFEr"} : (!fir.ref<!fir.box<!fir.array<2xi32>>>) -> (!fir.ref<!fir.box<!fir.array<2xi32>>>, !fir.ref<!fir.box<!fir.array<2xi32>>>)
! CHECK:               %[[VAL_16:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:               %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:               hlfir.assign %[[VAL_16]] to %[[VAL_19]] : i32, !fir.ref<i32>
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = arith.constant 0 : i32
! CHECK:               %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] : i32
! CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<!fir.box<!fir.array<2xi32>>>
! CHECK:               %[[VAL_24:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_23]] (%[[VAL_24]])  : (!fir.box<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:               hlfir.assign %[[VAL_22]] to %[[VAL_25]] : i32, !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
