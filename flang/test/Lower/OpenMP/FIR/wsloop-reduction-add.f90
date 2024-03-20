! RUN: bbc -emit-fir -hlfir=false -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL:   omp.declare_reduction @add_reduction_f64 : f64 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f64):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           omp.yield(%[[VAL_1]] : f64)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f64, %[[VAL_1:.*]]: f64):
! CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] fastmath<contract> : f64
! CHECK:           omp.yield(%[[VAL_2]] : f64)
! CHECK:         }

! CHECK-LABEL:   omp.declare_reduction @add_reduction_i64 : i64 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i64):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
! CHECK:           omp.yield(%[[VAL_1]] : i64)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64):
! CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i64
! CHECK:           omp.yield(%[[VAL_2]] : i64)
! CHECK:         }

! CHECK-LABEL:   omp.declare_reduction @add_reduction_f32 : f32 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f32):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           omp.yield(%[[VAL_1]] : f32)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32):
! CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] fastmath<contract> : f32
! CHECK:           omp.yield(%[[VAL_2]] : f32)
! CHECK:         }

! CHECK-LABEL:   omp.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
! CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:           omp.yield(%[[VAL_1]] : i32)

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
! CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
! CHECK:           omp.yield(%[[VAL_2]] : i32)
! CHECK:         }

! CHECK-LABEL:   func.func @_QPsimple_int_reduction() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_int_reductionEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_int_reductionEx"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_5:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_i32 %[[VAL_1]] -> %[[VAL_7:.*]] : !fir.ref<i32>)  for  (%[[VAL_8:.*]]) : i32 = (%[[VAL_4]]) to (%[[VAL_5]]) inclusive step (%[[VAL_6]]) {
! CHECK:               fir.store %[[VAL_8]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
! CHECK:               fir.store %[[VAL_11]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine simple_int_reduction
  integer :: x
  x = 0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = x + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine


! CHECK-LABEL:   func.func @_QPsimple_real_reduction() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_real_reductionEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFsimple_real_reductionEx"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_5:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_f32 %[[VAL_1]] -> %[[VAL_7:.*]] : !fir.ref<f32>)  for  (%[[VAL_8:.*]]) : i32 = (%[[VAL_4]]) to (%[[VAL_5]]) inclusive step (%[[VAL_6]]) {
! CHECK:               fir.store %[[VAL_8]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<f32>
! CHECK:               %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> f32
! CHECK:               %[[VAL_12:.*]] = arith.addf %[[VAL_9]], %[[VAL_11]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_12]] to %[[VAL_7]] : !fir.ref<f32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine simple_real_reduction
  real :: x
  x = 0.0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = x + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPsimple_int_reduction_switch_order() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_int_reduction_switch_orderEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_int_reduction_switch_orderEx"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_5:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_i32 %[[VAL_1]] -> %[[VAL_7:.*]] : !fir.ref<i32>)  for  (%[[VAL_8:.*]]) : i32 = (%[[VAL_4]]) to (%[[VAL_5]]) inclusive step (%[[VAL_6]]) {
! CHECK:               fir.store %[[VAL_8]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_10:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
! CHECK:               fir.store %[[VAL_11]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine simple_int_reduction_switch_order
  integer :: x
  x = 0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = i + x
  end do
  !$omp end do
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPsimple_real_reduction_switch_order() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_real_reduction_switch_orderEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFsimple_real_reduction_switch_orderEx"}
! CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_5:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_f32 %[[VAL_1]] -> %[[VAL_7:.*]] : !fir.ref<f32>)  for  (%[[VAL_8:.*]]) : i32 = (%[[VAL_4]]) to (%[[VAL_5]]) inclusive step (%[[VAL_6]]) {
! CHECK:               fir.store %[[VAL_8]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:               %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> f32
! CHECK:               %[[VAL_11:.*]] = fir.load %[[VAL_7]] : !fir.ref<f32>
! CHECK:               %[[VAL_12:.*]] = arith.addf %[[VAL_10]], %[[VAL_11]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_12]] to %[[VAL_7]] : !fir.ref<f32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine simple_real_reduction_switch_order
  real :: x
  x = 0.0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = i + x
  end do
  !$omp end do
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPmultiple_int_reductions_same_type() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_int_reductions_same_typeEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_int_reductions_same_typeEx"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFmultiple_int_reductions_same_typeEy"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFmultiple_int_reductions_same_typeEz"}
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_7:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_i32 %[[VAL_1]] -> %[[VAL_11:.*]] : !fir.ref<i32>, @add_reduction_i32 %[[VAL_2]] -> %[[VAL_12:.*]] : !fir.ref<i32>, @add_reduction_i32 %[[VAL_3]] -> %[[VAL_13:.*]] : !fir.ref<i32>)  for  (%[[VAL_14:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:               %[[VAL_16:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
! CHECK:               fir.store %[[VAL_17]] to %[[VAL_11]] : !fir.ref<i32>
! CHECK:               %[[VAL_18:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] : i32
! CHECK:               fir.store %[[VAL_20]] to %[[VAL_12]] : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
! CHECK:               fir.store %[[VAL_23]] to %[[VAL_13]] : !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine multiple_int_reductions_same_type
  integer :: x,y,z
  x = 0
  y = 0
  z = 0
  !$omp parallel
  !$omp do reduction(+:x,y,z)
  do i=1, 100
    x = x + i
    y = y + i
    z = z + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPmultiple_real_reductions_same_type() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_real_reductions_same_typeEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFmultiple_real_reductions_same_typeEx"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFmultiple_real_reductions_same_typeEy"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca f32 {bindc_name = "z", uniq_name = "_QFmultiple_real_reductions_same_typeEz"}
! CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<f32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_7:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_9:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_f32 %[[VAL_1]] -> %[[VAL_11:.*]] : !fir.ref<f32>, @add_reduction_f32 %[[VAL_2]] -> %[[VAL_12:.*]] : !fir.ref<f32>, @add_reduction_f32 %[[VAL_3]] -> %[[VAL_13:.*]] : !fir.ref<f32>)  for  (%[[VAL_14:.*]]) : i32 = (%[[VAL_8]]) to (%[[VAL_9]]) inclusive step (%[[VAL_10]]) {
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
! CHECK:               %[[VAL_16:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> f32
! CHECK:               %[[VAL_18:.*]] = arith.addf %[[VAL_15]], %[[VAL_17]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_18]] to %[[VAL_11]] : !fir.ref<f32>
! CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_12]] : !fir.ref<f32>
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> f32
! CHECK:               %[[VAL_22:.*]] = arith.addf %[[VAL_19]], %[[VAL_21]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_22]] to %[[VAL_12]] : !fir.ref<f32>
! CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
! CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> f32
! CHECK:               %[[VAL_26:.*]] = arith.addf %[[VAL_23]], %[[VAL_25]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_26]] to %[[VAL_13]] : !fir.ref<f32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine multiple_real_reductions_same_type
  real :: x,y,z
  x = 0.0
  y = 0.0
  z = 0.0
  !$omp parallel
  !$omp do reduction(+:x,y,z)
  do i=1, 100
    x = x + i
    y = y + i
    z = z + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPmultiple_reductions_different_type() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_reductions_different_typeEi"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca f64 {bindc_name = "w", uniq_name = "_QFmultiple_reductions_different_typeEw"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_reductions_different_typeEx"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca i64 {bindc_name = "y", uniq_name = "_QFmultiple_reductions_different_typeEy"}
! CHECK:           %[[VAL_4:.*]] = fir.alloca f32 {bindc_name = "z", uniq_name = "_QFmultiple_reductions_different_typeEz"}
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<i64>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_4]] : !fir.ref<f32>
! CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<f64>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_9:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_11:.*]] = arith.constant 100 : i32
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop reduction(@add_reduction_i32 %[[VAL_2]] -> %[[VAL_13:.*]] : !fir.ref<i32>, @add_reduction_i64 %[[VAL_3]] -> %[[VAL_14:.*]] : !fir.ref<i64>, @add_reduction_f32 %[[VAL_4]] -> %[[VAL_15:.*]] : !fir.ref<f32>, @add_reduction_f64 %[[VAL_1]] -> %[[VAL_16:.*]] : !fir.ref<f64>)  for  (%[[VAL_17:.*]]) : i32 = (%[[VAL_10]]) to (%[[VAL_11]]) inclusive step (%[[VAL_12]]) {
! CHECK:               fir.store %[[VAL_17]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_18:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] : i32
! CHECK:               fir.store %[[VAL_20]] to %[[VAL_13]] : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = fir.load %[[VAL_14]] : !fir.ref<i64>
! CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
! CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_23]] : i64
! CHECK:               fir.store %[[VAL_24]] to %[[VAL_14]] : !fir.ref<i64>
! CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
! CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> f32
! CHECK:               %[[VAL_28:.*]] = arith.addf %[[VAL_25]], %[[VAL_27]] fastmath<contract> : f32
! CHECK:               fir.store %[[VAL_28]] to %[[VAL_15]] : !fir.ref<f32>
! CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_16]] : !fir.ref<f64>
! CHECK:               %[[VAL_30:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> f64
! CHECK:               %[[VAL_32:.*]] = arith.addf %[[VAL_29]], %[[VAL_31]] fastmath<contract> : f64
! CHECK:               fir.store %[[VAL_32]] to %[[VAL_16]] : !fir.ref<f64>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }


subroutine multiple_reductions_different_type
  integer :: x
  integer(kind=8) :: y
  real :: z
  real(kind=8) :: w
  x = 0
  y = 0
  z = 0.0
  w = 0.0
  !$omp parallel
  !$omp do reduction(+:x,y,z,w)
  do i=1, 100
    x = x + i
    y = y + i
    z = z + i
    w = w + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine
