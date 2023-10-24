! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s 

!CHECK: func.func @_QPlastprivate_common() {
!CHECK: %[[val_0:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK: %[[val_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivate_commonEi"}
!CHECK: %[[val_2:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[val_3:.*]] = fir.convert %[[val_2]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_4:.*]] = fir.coordinate_of %[[val_3]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_5:.*]] = fir.convert %[[val_4]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK: %[[val_6:.*]] = fir.convert %[[val_2]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c4:.*]] = arith.constant 4 : index
!CHECK: %[[val_7:.*]] = fir.coordinate_of %[[val_6]], %[[val_c4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_8:.*]] = fir.convert %[[val_7]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK: %[[val_9:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivate_commonEx"}
!CHECK: %[[val_10:.*]] = fir.alloca f32 {bindc_name = "y", pinned, uniq_name = "_QFlastprivate_commonEy"}
!CHECK: %[[val_c1_i32:.*]] = arith.constant 1 : i32
!CHECK: %[[val_c100_i32:.*]] = arith.constant 100 : i32
!CHECK: %[[val_c1_i32_0:.*]] = arith.constant 1 : i32
!CHECK: omp.wsloop   for (%[[arg:.*]]) : i32 = (%[[val_c1_i32]]) to (%[[val_c100_i32]]) inclusive step (%[[val_c1_i32_0]]) {
!CHECK: fir.store %[[arg]] to %[[val_0]] : !fir.ref<i32>
!CHECK: %[[val_11:.*]] = arith.addi %[[arg]], %[[val_c1_i32_0]] : i32
!CHECK: %[[val_c0_i32:.*]] = arith.constant 0 : i32
!CHECK: %[[val_12:.*]] = arith.cmpi slt, %[[val_c1_i32_0]], %[[val_c0_i32]] : i32
!CHECK: %[[val_13:.*]] = arith.cmpi slt, %[[val_11]], %[[val_c100_i32]] : i32
!CHECK: %[[val_14:.*]] = arith.cmpi sgt, %[[val_11]], %[[val_c100_i32]] : i32
!CHECK: %[[val_15:.*]] = arith.select %[[val_12]], %[[val_13]], %[[val_14]] : i1
!CHECK: fir.if %[[val_15]] {
!CHECK: fir.store %[[val_11]] to %[[val_0]] : !fir.ref<i32>
!CHECK: %[[val_16:.*]] = fir.load %[[val_9]] : !fir.ref<f32>
!CHECK: fir.store %[[val_16]] to %[[val_5]] : !fir.ref<f32>
!CHECK: %[[val_17:.*]] = fir.load %[[val_10]] : !fir.ref<f32>
!CHECK: fir.store %[[val_17]] to %[[val_8]] : !fir.ref<f32>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: return
!CHECK: }
subroutine lastprivate_common
  common /c/ x, y
  real x, y
  !$omp do lastprivate(/c/)
  do i=1,100
  end do
  !$omp end do
end subroutine
