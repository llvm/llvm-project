! RUN: bbc -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: omp.reduction.declare
!CHECK-SAME: @[[RED_NAME:.*]] : !fir.logical<4> init {
!CHECK: ^bb0(%{{.*}}: !fir.logical<4>):
!CHECK:  %false = arith.constant false
!CHECK:  %[[false_fir:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK:  omp.yield(%[[false_fir]]  : !fir.logical<4>)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: !fir.logical<4>, %[[ARG1:.*]]: !fir.logical<4>):
!CHECK:  %[[arg0_i1:.*]] = fir.convert %[[ARG0]] : (!fir.logical<4>) -> i1
!CHECK:  %[[arg1_i1:.*]] = fir.convert %[[ARG1]] : (!fir.logical<4>) -> i1
!CHECK:  %[[RES:.*]] = arith.cmpi ne, %[[arg0_i1]], %[[arg1_i1]] : i1
!CHECK:  %[[RES_logical:.*]] = fir.convert %[[RES]] : (i1) -> !fir.logical<4>
!CHECK:  omp.yield(%[[RES_logical]] : !fir.logical<4>)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_reduction(
!CHECK-SAME: %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "y"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_reductionEi"}
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"}
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[XREF]] : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[CONVI_64:.*]] = fir.convert %[[I_PVT_VAL]] : (i32) -> i64
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      %[[SUBI:.*]] = arith.subi %[[CONVI_64]], %[[C1_64]] : i64
!CHECK:      %[[Y_PVT_REF:.*]] = fir.coordinate_of %[[ARRAY]], %[[SUBI]] : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[YVAL:.*]] = fir.load %[[Y_PVT_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[YVAL]], %[[XREF]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine simple_reduction(y)
  logical :: x, y(100)
  x = .true.
  !$omp parallel
  !$omp do reduction(.neqv.:x)
  do i=1, 100
    x = x .neqv. y(i)
  end do
  !$omp end do
  !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPsimple_reduction_switch_order(
!CHECK-SAME: %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "y"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_reduction_switch_orderEi"}
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFsimple_reduction_switch_orderEx"}
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[XREF]] : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[CONVI_64:.*]] = fir.convert %[[I_PVT_VAL]] : (i32) -> i64
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      %[[SUBI:.*]] = arith.subi %[[CONVI_64]], %[[C1_64]] : i64
!CHECK:      %[[Y_PVT_REF:.*]] = fir.coordinate_of %[[ARRAY]], %[[SUBI]] : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[YVAL:.*]] = fir.load %[[Y_PVT_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[YVAL]], %[[XREF]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine simple_reduction_switch_order(y)
  logical :: x, y(100)
  x = .true.
  !$omp parallel
  !$omp do reduction(.neqv.:x)
  do i=1, 100
  x = y(i) .neqv. x
  end do
  !$omp end do
  !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPmultiple_reductions
!CHECK-SAME %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "w"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_reductionsEi"}
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFmultiple_reductionsEx"}
!CHECK:  %[[YREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "y", uniq_name = "_QFmultiple_reductionsEy"}
!CHECK:  %[[ZREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z", uniq_name = "_QFmultiple_reductionsEz"}
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[XREF]] : !fir.ref<!fir.logical<4>>, @[[RED_NAME]] -> %[[YREF]] : !fir.ref<!fir.logical<4>>, @[[RED_NAME]] -> %[[ZREF]] : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[CONVI_64_1:.*]] = fir.convert %[[I_PVT_VAL1]] : (i32) -> i64
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      %[[SUBI_1:.*]] = arith.subi %[[CONVI_64_1]], %[[C1_64]] : i64
!CHECK:      %[[W_PVT_REF_1:.*]] = fir.coordinate_of %[[ARRAY]], %[[SUBI_1]] : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[WVAL:.*]] = fir.load %[[W_PVT_REF_1]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[WVAL]], %[[XREF]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      %[[I_PVT_VAL2:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[CONVI_64_2:.*]] = fir.convert %[[I_PVT_VAL2]] : (i32) -> i64
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      %[[SUBI_2:.*]] = arith.subi %[[CONVI_64_2]], %[[C1_64]] : i64
!CHECK:      %[[W_PVT_REF_2:.*]] = fir.coordinate_of %[[ARRAY]], %[[SUBI_2]] : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[WVAL:.*]] = fir.load %[[W_PVT_REF_2]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[WVAL]], %[[YREF]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      %[[I_PVT_VAL3:.*]] = fir.load %[[I_PVT_REF]] : !fir.ref<i32>
!CHECK:      %[[CONVI_64_3:.*]] = fir.convert %[[I_PVT_VAL3]] : (i32) -> i64
!CHECK:      %[[C1_64:.*]] = arith.constant 1 : i64
!CHECK:      %[[SUBI_3:.*]] = arith.subi %[[CONVI_64_3]], %[[C1_64]] : i64
!CHECK:      %[[W_PVT_REF_3:.*]] = fir.coordinate_of %[[ARRAY]], %[[SUBI_3]] : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[WVAL:.*]] = fir.load %[[W_PVT_REF_3]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[WVAL]], %[[ZREF]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine multiple_reductions(w)
  logical :: x,y,z,w(100)
  x = .true.
  y = .true.
  z = .true.
  !$omp parallel
  !$omp do reduction(.neqv.:x,y,z)
  do i=1, 100
  x = x .neqv. w(i)
  y = y .neqv. w(i)
  z = z .neqv. w(i)
  end do
  !$omp end do
  !$omp end parallel
end subroutine
