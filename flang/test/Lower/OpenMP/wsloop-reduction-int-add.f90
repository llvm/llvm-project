! RUN: bbc -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

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
!CHECK-SAME: @[[RED_I32_NAME:.*]] : i32 init {
!CHECK: ^bb0(%{{.*}}: i32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:  omp.yield(%[[C0_1]] : i32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:  %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:  omp.yield(%[[RES]] : i32)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_reduction
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"}
!CHECK:  %[[C0_2:.*]] = arith.constant 0 : i32
!CHECK:  fir.store %[[C0_2]] to %[[XREF]] : !fir.ref<i32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[XREF]] : !fir.ref<i32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL]], %[[XREF]] : !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return

subroutine simple_reduction
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

!CHECK-LABEL: func.func @_QPsimple_reduction_switch_order
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsimple_reduction_switch_orderEx"}
!CHECK:  %[[C0_2:.*]] = arith.constant 0 : i32
!CHECK:  fir.store %[[C0_2]] to %[[XREF]] : !fir.ref<i32>
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[XREF]] : !fir.ref<i32>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]])
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL]], %[[XREF]] : !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return

subroutine simple_reduction_switch_order
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

!CHECK-LABEL: func.func @_QPmultiple_reductions_same_type
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_reductions_same_typeEx"}
!CHECK:  %[[YREF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFmultiple_reductions_same_typeEy"}
!CHECK:  %[[ZREF:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFmultiple_reductions_same_typeEz"}
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[XREF]] : !fir.ref<i32>, @[[RED_I32_NAME]] -> %[[YREF]] : !fir.ref<i32>, @[[RED_I32_NAME]] -> %[[ZREF]] : !fir.ref<i32>) for  (%[[IVAL]]) : i32
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL1]], %[[XREF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL2:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL2]], %[[YREF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL3:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      omp.reduction %[[I_PVT_VAL3]], %[[ZREF]] : !fir.ref<i32>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return

subroutine multiple_reductions_same_type
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

!CHECK-LABEL: func.func @_QPmultiple_reductions_different_type
!CHECK:  %[[XREF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_reductions_different_typeEx"}
!CHECK:  %[[YREF:.*]] = fir.alloca i64 {bindc_name = "y", uniq_name = "_QFmultiple_reductions_different_typeEy"}
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    omp.wsloop   reduction(@[[RED_I32_NAME]] -> %[[XREF]] : !fir.ref<i32>, @[[RED_I64_NAME]] -> %[[YREF]] : !fir.ref<i64>) for  (%[[IVAL:.*]]) : i32
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[C1_32:.*]] = arith.constant 1 : i32
!CHECK:      omp.reduction %[[C1_32]], %[[XREF]] : !fir.ref<i32>
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      omp.reduction %[[C1_64]], %[[YREF]] : !fir.ref<i64>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return

subroutine multiple_reductions_different_type
  integer :: x
  integer(kind=8) :: y
  !$omp parallel
  !$omp do reduction(+:x,y)
  do i=1, 100
    x = x + 1_4
    y = y + 1_8
  end do
  !$omp end do
  !$omp end parallel
end subroutine
