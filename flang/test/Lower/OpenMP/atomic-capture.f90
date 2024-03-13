! This test checks the lowering of atomic capture

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s 



program OmpAtomicCapture
    use omp_lib                                                                                                       

!CHECK: %[[VAL_X_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAL_X_DECLARE:.*]]:2 = hlfir.declare %[[VAL_X_ALLOCA]] {{.*}}
!CHECK: %[[VAL_Y_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[VAL_Y_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Y_ALLOCA]] {{.*}}
    integer :: x, y

!CHECK: %[[VAL_Y_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK: omp.atomic.capture hint(uncontended) {
!CHECK: omp.atomic.update %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.muli %[[VAL_Y_LOADED]], %[[ARG]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
!CHECK: omp.atomic.read %[[VAL_X_DECLARE]]#1 = %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32>, i32
!CHECK: }
    !$omp atomic hint(omp_sync_hint_uncontended) capture
        y = x * y 
        x = y
    !$omp end atomic

!CHECK: %[[VAL_20:.*]] = arith.constant 20 : i32
!CHECK: %[[VAL_8:.*]] = arith.constant 8 : i32
!CHECK: %[[VAL_X_LOADED:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
!CHECK: %[[SUB:.*]] = arith.subi %[[VAL_8]], %[[VAL_X_LOADED]] : i32
!CHECK: %[[NO_REASSOC:.*]] = hlfir.no_reassoc %[[SUB]] : i32
!CHECK: %[[ADD:.*]] = arith.addi  %[[VAL_20]], %[[NO_REASSOC]] : i32
!CHECK: omp.atomic.capture memory_order(acquire) hint(nonspeculative) {
!CHECK:   omp.atomic.read %[[VAL_X_DECLARE]]#1 = %[[VAL_Y_DECLARE]]#1 : !fir.ref<i32>, i32
!CHECK:   omp.atomic.write %[[VAL_Y_DECLARE]]#1 = %[[ADD]] : !fir.ref<i32>, i32
!CHECK: }
!CHECK: return
!CHECK: }
    !$omp atomic hint(omp_lock_hint_nonspeculative) capture acquire
        x = y
        y = 2 * 10 + (8 - x) 
    !$omp end atomic
end program


!CHECK: func.func @_QPpointers_in_atomic_capture() {
subroutine pointers_in_atomic_capture()

!CHECK: %[[VAL_A_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFpointers_in_atomic_captureEa"}
!CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_A_ALLOCA]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_A_DECLARE:.*]]:2 = hlfir.declare %[[VAL_A_ALLOCA]] {{.*}}
!CHECK: %[[VAL_B_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFpointers_in_atomic_captureEb"}
!CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_B_ALLOCA]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_B_DECLARE:.*]]:2 = hlfir.declare %[[VAL_B_ALLOCA]] {{.*}}
!CHECK: %[[VAL_C_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "c", fir.target, uniq_name = "_QFpointers_in_atomic_captureEc"}
!CHECK: %[[VAL_C_DECLARE:.*]]:2 = hlfir.declare %[[VAL_C_ALLOCA]] {{.*}}
!CHECK: %[[VAL_D_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "d", fir.target, uniq_name = "_QFpointers_in_atomic_captureEd"}
!CHECK: %[[VAL_D_DECLARE:.*]]:2 = hlfir.declare %[[VAL_D_ALLOCA]] {{.*}}
    integer, pointer :: a, b
    integer, target :: c, d

!CHECK: %[[EMBOX:.*]] = fir.embox %[[VAL_C_DECLARE]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_A_DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[EMBOX:.*]] = fir.embox %[[VAL_D_DECLARE]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store %[[EMBOX]] to %[[VAL_B_DECLARE]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    a=>c
    b=>d

!CHECK: %[[VAL_A_LOADED:.*]] = fir.load %[[VAL_A_DECLARE]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_A_BOX_ADDR:.*]] = fir.box_addr %[[VAL_A_LOADED]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[VAL_B_LOADED:.*]] = fir.load %[[VAL_B_DECLARE]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_B_BOX_ADDR:.*]] = fir.box_addr %[[VAL_B_LOADED]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[VAL_B_LOADED_2:.*]] = fir.load %[[VAL_B_DECLARE]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[VAL_B_BOX_ADDR_2:.*]] = fir.box_addr %[[VAL_B_LOADED_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[VAL_B:.*]] = fir.load %[[VAL_B_BOX_ADDR_2]] : !fir.ptr<i32>
!CHECK: omp.atomic.capture {
!CHECK: omp.atomic.update %[[VAL_A_BOX_ADDR]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[TEMP:.*]] = arith.addi %[[ARG]], %[[VAL_B]] : i32
!CHECK: omp.yield(%[[TEMP]] : i32)
!CHECK: }
!CHECK: omp.atomic.read %[[VAL_B_BOX_ADDR]] = %[[VAL_A_BOX_ADDR]] : !fir.ptr<i32>, i32
!CHECK: }
!CHECK: return
!CHECK: }
    !$omp atomic capture
        a = a + b
        b = a
    !$omp end atomic
end subroutine

!CHECK-LABEL: func.func @_QPreal_to_double() {
subroutine real_to_double()
    !CHECK:     %[[C_ALLOC:.*]] = fir.alloca f64
    !CHECK:     %[[C_REF:.*]] = fir.alloca f32 {bindc_name = "c", uniq_name = "_QFreal_to_doubleEc"}
    !CHECK:     %[[C_DECL:.*]]:2 = hlfir.declare %[[C_REF]] {uniq_name = "_QFreal_to_doubleEc"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
    !CHECK:     %[[C2_ALLOC:.*]] = fir.alloca f64 {bindc_name = "c2", uniq_name = "_QFreal_to_doubleEc2"}
    !CHECK:     %[[C2_DECL:.*]]:2 = hlfir.declare %[[C2_ALLOC]] {uniq_name = "_QFreal_to_doubleEc2"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
        real :: c
        double precision :: c2
    !CHECK:     %[[C_LOAD:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<f32>
    !CHECK:     %[[C_CVT:.*]] = fir.convert %[[C_LOAD]] : (f32) -> f64
    !CHECK:     fir.store %[[C_CVT]] to %[[C_ALLOC]] : !fir.ref<f64>
    !CHECK:     %cst = arith.constant 2.000000e+00 : f32
    !CHECK:     omp.atomic.capture {
    !CHECK:       omp.atomic.read %[[C2_DECL]]#1 = %[[C_ALLOC]] : !fir.ref<f64>, f32
    !CHECK:       omp.atomic.update %[[C_ALLOC]] : !fir.ref<f64> {
    !CHECK:       ^bb0(%arg0: f32):
    !CHECK:         %[[RES:.*]] = arith.mulf %cst, %arg0 fastmath<contract> : f32
    !CHECK:         omp.yield(%[[RES]] : f32)
    !CHECK:       }
    !CHECK:     }
    !CHECK:     return
    !CHECK:   }
    !$omp atomic capture
        c2 = c
        c = 2.0 * c
    !$omp end atomic
    end
