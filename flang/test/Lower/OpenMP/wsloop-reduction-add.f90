! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_F64_NAME:.*]] : f64 init {
!CHECK: ^bb0(%{{.*}}: f64):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:  omp.yield(%[[C0_1]] : f64)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64):
!CHECK:  %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}}: f64
!CHECK:  omp.yield(%[[RES]] : f64)
!CHECK: }

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_I64_NAME:.*]] : i64 init {
!CHECK: ^bb0(%{{.*}}: i64):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i64
!CHECK:  omp.yield(%[[C0_1]] : i64)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64):
!CHECK:  %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i64
!CHECK:  omp.yield(%[[RES]] : i64)
!CHECK: }

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_F32_NAME:.*]] : f32 init {
!CHECK: ^bb0(%{{.*}}: f32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  omp.yield(%[[C0_1]] : f32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
!CHECK:  %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}}: f32
!CHECK:  omp.yield(%[[RES]] : f32)
!CHECK: }

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_I32_NAME:.*]] : i32 init {
!CHECK: ^bb0(%{{.*}}: i32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:  omp.yield(%[[C0_1]] : i32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:  %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:  omp.yield(%[[RES]] : i32)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_int_reduction
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_int_reductionEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFsimple_int_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[C0_2:.*]] = arith.constant 0 : i32
!CHECK:  hlfir.assign %[[C0_2]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_int_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<i32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPsimple_real_reduction
!CHECK:  %[[XREF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFsimple_real_reductionEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFsimple_real_reductionEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[C0_2:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  hlfir.assign %[[C0_2]] to %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_real_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_F32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<f32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL_i32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL_f32:.*]] = fir.convert %[[I_PVT_VAL_i32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL_f32]], %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPsimple_int_reduction_switch_order
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_int_reduction_switch_orderEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %2 {uniq_name = "_QFsimple_int_reduction_switch_orderEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[C0_2:.*]] = arith.constant 0 : i32
!CHECK:  hlfir.assign %[[C0_2]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_int_reduction_switch_orderEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<i32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPsimple_real_reduction_switch_order
!CHECK:  %[[XREF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFsimple_real_reduction_switch_orderEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFsimple_real_reduction_switch_orderEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[C0_2:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:  hlfir.assign %[[C0_2]] to %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_real_reduction_switch_orderEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_F32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<f32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL_i32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL_f32:.*]] = fir.convert %[[I_PVT_VAL_i32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL_f32]], %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPmultiple_int_reductions_same_type
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_int_reductions_same_typeEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFmultiple_int_reductions_same_typeEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[YREF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFmultiple_int_reductions_same_typeEy"}
!CHECK:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[YREF]] {uniq_name = "_QFmultiple_int_reductions_same_typeEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[ZREF:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFmultiple_int_reductions_same_typeEz"}
!CHECK:  %[[Z_DECL:.*]]:2 = hlfir.declare %[[ZREF]] {uniq_name = "_QFmultiple_int_reductions_same_typeEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFmultiple_int_reductions_same_typeEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<i32>, @[[RED_I32_NAME]] -> %[[Y_DECL]]#0 : !fir.ref<i32>, @[[RED_I32_NAME]] -> %[[Z_DECL]]#0 : !fir.ref<i32>) for  (%[[IVAL]]) : i32
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL1]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL2:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL2]], %[[Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL3:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL3]], %[[Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPmultiple_real_reductions_same_type
!CHECK:  %[[XREF:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFmultiple_real_reductions_same_typeEx"}
!CHECK:  %[[X_DECL]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFmultiple_real_reductions_same_typeEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[YREF:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFmultiple_real_reductions_same_typeEy"}
!CHECK:  %[[Y_DECL]]:2 = hlfir.declare %[[YREF]] {uniq_name = "_QFmultiple_real_reductions_same_typeEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  %[[ZREF:.*]] = fir.alloca f32 {bindc_name = "z", uniq_name = "_QFmultiple_real_reductions_same_typeEz"}
!CHECK:  %[[Z_DECL]]:2 = hlfir.declare %[[ZREF]] {uniq_name = "_QFmultiple_real_reductions_same_typeEz"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFmultiple_real_reductions_same_typeEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.wsloop   reduction(@[[RED_F32_NAME]] -> %[[X_DECL]]#0 : !fir.ref<f32>, @[[RED_F32_NAME]] -> %[[Y_DECL]]#0 : !fir.ref<f32>, @[[RED_F32_NAME]] -> %[[Z_DECL]]#0 : !fir.ref<f32>) for  (%[[IVAL]]) : i32
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1_F32:.*]] = fir.convert %[[I_PVT_VAL1_I32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL1_F32]], %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      %[[I_PVT_VAL2_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL2_F32:.*]] = fir.convert %[[I_PVT_VAL2_I32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL2_F32]], %[[Y_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      %[[I_PVT_VAL3_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL3_F32:.*]] = fir.convert %[[I_PVT_VAL3_I32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL3_F32]], %[[Z_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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

!CHECK-LABEL: func.func @_QPmultiple_reductions_different_type
!CHECK:  %[[WREF:.*]] = fir.alloca f64 {bindc_name = "w", uniq_name = "_QFmultiple_reductions_different_typeEw"}
!CHECK:  %[[W_DECL:.*]]:2 = hlfir.declare %[[WREF]] {uniq_name = "_QFmultiple_reductions_different_typeEw"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_reductions_different_typeEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFmultiple_reductions_different_typeEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[YREF:.*]] = fir.alloca i64 {bindc_name = "y", uniq_name = "_QFmultiple_reductions_different_typeEy"}
!CHECK:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[YREF]] {uniq_name = "_QFmultiple_reductions_different_typeEy"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:  %[[ZREF:.*]] = fir.alloca f32 {bindc_name = "z", uniq_name = "_QFmultiple_reductions_different_typeEz"}
!CHECK:  %[[Z_DECL:.*]]:2 = hlfir.declare %[[ZREF]] {uniq_name = "_QFmultiple_reductions_different_typeEz"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFmultiple_reductions_different_typeEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[X_DECL]]#0  : !fir.ref<i32>, @[[RED_I64_NAME]] -> %[[Y_DECL]]#0 : !fir.ref<i64>, @[[RED_F32_NAME]] -> %[[Z_DECL]]#0  : !fir.ref<f32>, @[[RED_F64_NAME]] -> %[[W_DECL]]#0  : !fir.ref<f64>) for  (%[[IVAL:.*]]) : i32
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL1_I32]], %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL2_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL2_I64:.*]] = fir.convert %[[I_PVT_VAL2_I32]] : (i32) -> i64
!CHECK:      omp.reduction %[[I_PVT_VAL2_I64]], %[[Y_DECL]]#0 : i64, !fir.ref<i64>
!CHECK:      %[[I_PVT_VAL3_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL3_F32:.*]] = fir.convert %[[I_PVT_VAL3_I32]] : (i32) -> f32
!CHECK:      omp.reduction %[[I_PVT_VAL3_F32]], %[[Z_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:      %[[I_PVT_VAL4_I32:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL4_F64:.*]] = fir.convert %[[I_PVT_VAL4_I32]] : (i32) -> f64
!CHECK:      omp.reduction %[[I_PVT_VAL4_F64]], %[[W_DECL]]#0 : f64, !fir.ref<f64>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
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
