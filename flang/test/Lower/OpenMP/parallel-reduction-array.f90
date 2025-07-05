! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s --check-prefix=CPU
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s --check-prefix=CPU

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-is-target-device -fopenmp-is-gpu -o - %s 2>&1 | FileCheck %s --check-prefix=GPU
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device -o - %s 2>&1 | \
! RUN: FileCheck %s --check-prefix=GPU %}

program reduce
integer, dimension(3) :: i = 0

!$omp parallel reduction(+:i)
i(1) = 1
i(2) = 2
i(3) = 3
!$omp end parallel

print *,i
end program

! CPU-LABEL:   omp.declare_reduction @add_reduction_byref_box_3xi32 : !fir.ref<!fir.box<!fir.array<3xi32>>> alloc {
! CPU:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CPU:           omp.yield(%[[VAL_8]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CPU-LABEL:   } init {
! CPU:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>, %[[ALLOC:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CPU:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CPU:           %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           %[[VAL_4:.*]] = arith.constant 3 : index
! CPU:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CPU:           %[[VAL_1:.*]] = fir.allocmem !fir.array<3xi32> {bindc_name = ".tmp", uniq_name = ""}
! CPU:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_5]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<3xi32>>,
! CPU:           %[[TRUE:.*]]  = arith.constant true
!fir.shape<1>) -> (!fir.heap<!fir.array<3xi32>>, !fir.heap<!fir.array<3xi32>>)
! CPU:           %[[C0:.*]] = arith.constant 0 : index
! CPU:           %[[DIMS:.*]]:3 = fir.box_dims %[[VAL_3]], %[[C0]] : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
! CPU:           %[[SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CPU:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[SHIFT]]) : (!fir.heap<!fir.array<3xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<3xi32>>
! CPU:           hlfir.assign %[[VAL_2]] to %[[VAL_7]] : i32, !fir.box<!fir.array<3xi32>>
! CPU:           fir.store %[[VAL_7]] to %[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           omp.yield(%[[ALLOC]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CPU:         } combiner {
! CPU:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CPU:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           %[[C1:.*]] = arith.constant 1 : index
! CPU:           %[[C3:.*]] = arith.constant 3 : index
! CPU:           %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C1]], %[[C3]] : (index, index) -> !fir.shapeshift<1>
! CPU:           %[[C1_0:.*]] = arith.constant 1 : index
! CPU:           fir.do_loop %[[VAL_8:.*]] = %[[C1_0]] to %[[C3]] step %[[C1_0]] unordered {
! CPU:             %[[VAL_9:.*]] = fir.array_coor %[[VAL_2]](%[[SHAPE_SHIFT]]) %[[VAL_8]] : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CPU:             %[[VAL_10:.*]] = fir.array_coor %[[VAL_3]](%[[SHAPE_SHIFT]]) %[[VAL_8]] : (!fir.box<!fir.array<3xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CPU:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CPU:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CPU:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CPU:             fir.store %[[VAL_13]] to %[[VAL_9]] : !fir.ref<i32>
! CPU:           }
! CPU:           omp.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CPU:         }  cleanup {
! CPU:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.array<3xi32>>>):
! CPU:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<3xi32>>) -> !fir.ref<!fir.array<3xi32>>
! CPU:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3xi32>>) -> i64
! CPU:           %[[VAL_4:.*]] = arith.constant 0 : i64
! CPU:           %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CPU:           fir.if %[[VAL_5]] {
! CPU:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<3xi32>>) -> !fir.heap<!fir.array<3xi32>>
! CPU:             fir.freemem %[[VAL_6]] : !fir.heap<!fir.array<3xi32>>
! CPU:           }
! CPU:           omp.yield
! CPU:         }

! CPU-LABEL:   func.func @_QQmain()
! CPU:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<!fir.array<3xi32>>
! CPU:           %[[VAL_1:.*]] = arith.constant 3 : index
! CPU:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CPU:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "_QFEi"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CPU:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CPU:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.array<3xi32>>
! CPU:           fir.store %[[VAL_4]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:           omp.parallel reduction(byref @add_reduction_byref_box_3xi32 %[[VAL_5]] -> %[[VAL_6:.*]] : !fir.ref<!fir.box<!fir.array<3xi32>>>) {
! CPU:             %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFEi"} : (!fir.ref<!fir.box<!fir.array<3xi32>>>) -> (!fir.ref<!fir.box<!fir.array<3xi32>>>, !fir.ref<!fir.box<!fir.array<3xi32>>>)
! CPU:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CPU:             %[[VAL_9:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:             %[[VAL_10:.*]] = arith.constant 1 : index
! CPU:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_10]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CPU:             hlfir.assign %[[VAL_8]] to %[[VAL_11]] : i32, !fir.ref<i32>
! CPU:             %[[VAL_12:.*]] = arith.constant 2 : i32
! CPU:             %[[VAL_13:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:             %[[VAL_14:.*]] = arith.constant 2 : index
! CPU:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_14]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CPU:             hlfir.assign %[[VAL_12]] to %[[VAL_15]] : i32, !fir.ref<i32>
! CPU:             %[[VAL_16:.*]] = arith.constant 3 : i32
! CPU:             %[[VAL_17:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.array<3xi32>>>
! CPU:             %[[VAL_18:.*]] = arith.constant 3 : index
! CPU:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_18]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CPU:             hlfir.assign %[[VAL_16]] to %[[VAL_19]] : i32, !fir.ref<i32>
! CPU:             omp.terminator
! CPU:           }

! GPU: omp.declare_reduction {{.*}} alloc {
! GPU: } init {
! GPU-NOT: fir.allocmem {{.*}} {bindc_name = ".tmp", {{.*}}}
! GPU:     fir.alloca {{.*}} {bindc_name = ".tmp"}
! GPU: } combiner {
! GPU: }
