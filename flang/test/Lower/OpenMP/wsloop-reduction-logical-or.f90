! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

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
!CHECK:  %[[RES:.*]] = arith.ori %[[arg0_i1]], %[[arg1_i1]] : i1
!CHECK:  %[[RES_logical:.*]] = fir.convert %[[RES]] : (i1) -> !fir.logical<4>
!CHECK:  omp.yield(%[[RES_logical]] : !fir.logical<4>)
!CHECK: }

!CHECK-LABEL: func.func @_QPsimple_reduction(
!CHECK-SAME: %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "y"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_reductionEi"}
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFsimple_reductionEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARRAY]](%4) {uniq_name = "_QFsimple_reductionEy"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[X_DECL]]#0 : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[I_PVT_64:.*]] = fir.convert %[[I_PVT]] : (i32) -> i64
!CHECK:      %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 (%[[I_PVT_64]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[Y_I_VAL:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[Y_I_VAL]], %[[X_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine simple_reduction(y)
  logical :: x, y(100)
  x = .true.
  !$omp parallel
  !$omp do reduction(.or.:x)
  do i=1, 100
    x = x .or. y(i)
  end do
  !$omp end do
  !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPsimple_reduction_switch_order(
!CHECK-SAME: %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "y"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_reduction_switch_orderEi"}
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFsimple_reduction_switch_orderEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFsimple_reduction_switch_orderEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARRAY]](%{{.*}}) {uniq_name = "_QFsimple_reduction_switch_orderEy"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFsimple_reduction_switch_orderEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[X_DECL]]#0 : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[CONVI_64:.*]] = fir.convert %[[I_PVT_VAL]] : (i32) -> i64
!CHECK:      %[[Y_I_REF:.*]] = hlfir.designate %[[Y_DECL]]#0 (%[[CONVI_64]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[YVAL:.*]] = fir.load %[[Y_I_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[YVAL]], %[[X_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine simple_reduction_switch_order(y)
  logical :: x, y(100)
  x = .true.
  !$omp parallel
  !$omp do reduction(.or.:x)
  do i=1, 100
  x = y(i) .or. x
  end do
  !$omp end do
  !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPmultiple_reductions
!CHECK-SAME %[[ARRAY:.*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "w"}) {
!CHECK:  %[[IREF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_reductionsEi"}
!CHECK:  %[[I_DECL:.*]]:2 = hlfir.declare %[[IREF]] {uniq_name = "_QFmultiple_reductionsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[W_DECL:.*]]:2 = hlfir.declare %[[ARRAY]](%{{.*}}) {uniq_name = "_QFmultiple_reductionsEw"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
!CHECK:  %[[XREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFmultiple_reductionsEx"}
!CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[XREF]] {uniq_name = "_QFmultiple_reductionsEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:  %[[YREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "y", uniq_name = "_QFmultiple_reductionsEy"}
!CHECK:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[YREF]] {uniq_name = "_QFmultiple_reductionsEy"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:  %[[ZREF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "z", uniq_name = "_QFmultiple_reductionsEz"}
!CHECK:  %[[Z_DECL:.*]]:2 = hlfir.declare %[[ZREF]] {uniq_name = "_QFmultiple_reductionsEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:  omp.parallel
!CHECK:    %[[I_PVT_REF:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFmultiple_reductionsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C1_1:.*]] = arith.constant 1 : i32
!CHECK:    %[[C100:.*]] = arith.constant 100 : i32
!CHECK:    %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:    omp.wsloop   reduction(@[[RED_NAME]] -> %[[X_DECL]]#0 : !fir.ref<!fir.logical<4>>, @[[RED_NAME]] -> %[[Y_DECL]]#0 :
!!fir.ref<!fir.logical<4>>, @[[RED_NAME]] -> %[[Z_DECL]]#0 : !fir.ref<!fir.logical<4>>) for  (%[[IVAL:.*]]) : i32 = (%[[C1_1]]) to (%[[C100]]) inclusive step (%[[C1_2]]) {
!CHECK:      fir.store %[[IVAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      %[[I_PVT_VAL1:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[CONVI_64_1:.*]] = fir.convert %[[I_PVT_VAL1]] : (i32) -> i64
!CHECK:      %[[W_I_REF:.*]] = hlfir.designate %[[W_DECL]]#0 (%[[CONVI_64_1]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[W_I_VAL:.*]] = fir.load %[[W_I_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[W_I_VAL]], %[[X_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      %[[I_PVT_VAL2:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[CONVI_64_2:.*]] = fir.convert %[[I_PVT_VAL2]] : (i32) -> i64
!CHECK:      %[[W_I_REF:.*]] = hlfir.designate %[[W_DECL]]#0 (%[[CONVI_64_2]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[W_I_VAL:.*]] = fir.load %[[W_I_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[W_I_VAL]], %[[Y_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      %[[I_PVT_VAL2:.*]] = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      %[[CONVI_64_2:.*]] = fir.convert %[[I_PVT_VAL2]] : (i32) -> i64
!CHECK:      %[[W_I_REF:.*]] = hlfir.designate %[[W_DECL]]#0 (%[[CONVI_64_2]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
!CHECK:      %[[W_I_VAL:.*]] = fir.load %[[W_I_REF]] : !fir.ref<!fir.logical<4>>
!CHECK:      omp.reduction %[[W_I_VAL]], %[[Z_DECL]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:      omp.yield
!CHECK:    omp.terminator
!CHECK:  return
subroutine multiple_reductions(w)
  logical :: x,y,z,w(100)
  x = .true.
  y = .true.
  z = .true.
  !$omp parallel
  !$omp do reduction(.or.:x,y,z)
  do i=1, 100
  x = x .or. w(i)
  y = y .or. w(i)
  z = z .or. w(i)
  end do
  !$omp end do
  !$omp end parallel
end subroutine

