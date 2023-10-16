! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: func.func @_QPfirstprivate_common() {
!CHECK: %[[val_0:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[val_1:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_2:.*]] = fir.coordinate_of %[[val_1]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_3:.*]] = fir.convert %[[val_2]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK: %[[VAL_3_DECL:.*]]:2 = hlfir.declare %[[val_3]] {uniq_name = "_QFfirstprivate_commonEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[val_4:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c4:.*]] = arith.constant 4 : index
!CHECK: %[[val_5:.*]] = fir.coordinate_of %[[val_4]], %[[val_c4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_6:.*]] = fir.convert %[[val_5]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK: %[[VAL_6_DECL:.*]]:2 = hlfir.declare %[[val_6]] {uniq_name = "_QFfirstprivate_commonEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: omp.parallel {
!CHECK: %[[val_7:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFfirstprivate_commonEx"}
!CHECK: %[[VAL_7_DECL:.*]]:2 = hlfir.declare %[[val_7]] {uniq_name = "_QFfirstprivate_commonEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[val_8:.*]] = fir.load %[[VAL_3_DECL]]#1 : !fir.ref<f32>
!CHECK: fir.store %[[val_8]] to %[[VAL_7_DECL]]#1 : !fir.ref<f32>
!CHECK: %[[val_9:.*]] = fir.alloca f32 {bindc_name = "y", pinned, uniq_name = "_QFfirstprivate_commonEy"}
!CHECK: %[[VAL_9_DECL:.*]]:2 = hlfir.declare %[[val_9]] {uniq_name = "_QFfirstprivate_commonEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[val_10:.*]] = fir.load %[[VAL_6_DECL]]#1 : !fir.ref<f32>
!CHECK: fir.store %[[val_10]] to %[[VAL_9_DECL]]#1 : !fir.ref<f32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }

subroutine firstprivate_common
  common /c/ x, y
  real x, y
  !$omp parallel firstprivate(/c/)
  !$omp end parallel
end subroutine
