! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine sectionsReduction(x,y)
  real :: x, y

  !$omp parallel
    !$omp sections reduction(+:x,y)
        x = x + 1
        y = x
      !$omp section
        x = x + 2
        y = x
    !$omp end sections
  !$omp end parallel

  !$omp parallel sections reduction(+:x) reduction(+:y)
      x = x + 1
      y = x
    !$omp section
      x = x + 2
      y = x
  !$omp end parallel sections
end subroutine

! CHECK-LABEL:   omp.declare_reduction @add_reduction_f32 : f32 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f32):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           omp.yield(%[[VAL_1]] : f32)
! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32):
! CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] fastmath<contract> : f32
! CHECK:           omp.yield(%[[VAL_2]] : f32)
! CHECK:         }

! CHECK-LABEL:   func.func @_QPsectionsreduction(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFsectionsreductionEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           omp.parallel {
! CHECK:             omp.sections reduction(@add_reduction_f32 %[[VAL_3]]#0 -> %[[VAL_5:.*]], @add_reduction_f32 %[[VAL_4]]#0 -> %[[VAL_6:.*]] : !fir.ref<f32>, !fir.ref<f32>) {
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_7:.*]]: !fir.ref<f32>, %[[VAL_8:.*]]: !fir.ref<f32>):
! CHECK:                 %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFsectionsreductionEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_11:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<f32>
! CHECK:                 %[[VAL_12:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:                 %[[VAL_13:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] fastmath<contract> : f32
! CHECK:                 hlfir.assign %[[VAL_13]] to %[[VAL_9]]#0 : f32, !fir.ref<f32>
! CHECK:                 %[[VAL_14:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<f32>
! CHECK:                 hlfir.assign %[[VAL_14]] to %[[VAL_10]]#0 : f32, !fir.ref<f32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_15:.*]]: !fir.ref<f32>, %[[VAL_16:.*]]: !fir.ref<f32>):
! CHECK:                 %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_15]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QFsectionsreductionEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_19:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<f32>
! CHECK:                 %[[VAL_20:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:                 %[[VAL_21:.*]] = arith.addf %[[VAL_19]], %[[VAL_20]] fastmath<contract> : f32
! CHECK:                 hlfir.assign %[[VAL_21]] to %[[VAL_17]]#0 : f32, !fir.ref<f32>
! CHECK:                 %[[VAL_22:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<f32>
! CHECK:                 hlfir.assign %[[VAL_22]] to %[[VAL_18]]#0 : f32, !fir.ref<f32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.parallel {
! CHECK:             omp.sections reduction(@add_reduction_f32 %[[VAL_3]]#0 -> %[[VAL_23:.*]], @add_reduction_f32 %[[VAL_4]]#0 -> %[[VAL_24:.*]] : !fir.ref<f32>, !fir.ref<f32>) {
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_25:.*]]: !fir.ref<f32>, %[[VAL_26:.*]]: !fir.ref<f32>):
! CHECK:                 %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_25]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_26]] {uniq_name = "_QFsectionsreductionEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_29:.*]] = fir.load %[[VAL_27]]#0 : !fir.ref<f32>
! CHECK:                 %[[VAL_30:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:                 %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[VAL_30]] fastmath<contract> : f32
! CHECK:                 hlfir.assign %[[VAL_31]] to %[[VAL_27]]#0 : f32, !fir.ref<f32>
! CHECK:                 %[[VAL_32:.*]] = fir.load %[[VAL_27]]#0 : !fir.ref<f32>
! CHECK:                 hlfir.assign %[[VAL_32]] to %[[VAL_28]]#0 : f32, !fir.ref<f32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:               ^bb0(%[[VAL_33:.*]]: !fir.ref<f32>, %[[VAL_34:.*]]: !fir.ref<f32>):
! CHECK:                 %[[VAL_35:.*]]:2 = hlfir.declare %[[VAL_33]] {uniq_name = "_QFsectionsreductionEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_36:.*]]:2 = hlfir.declare %[[VAL_34]] {uniq_name = "_QFsectionsreductionEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:                 %[[VAL_37:.*]] = fir.load %[[VAL_35]]#0 : !fir.ref<f32>
! CHECK:                 %[[VAL_38:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:                 %[[VAL_39:.*]] = arith.addf %[[VAL_37]], %[[VAL_38]] fastmath<contract> : f32
! CHECK:                 hlfir.assign %[[VAL_39]] to %[[VAL_35]]#0 : f32, !fir.ref<f32>
! CHECK:                 %[[VAL_40:.*]] = fir.load %[[VAL_35]]#0 : !fir.ref<f32>
! CHECK:                 hlfir.assign %[[VAL_40]] to %[[VAL_36]]#0 : f32, !fir.ref<f32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
