! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

! This test checks the lowering of atomic capture

program OmpAtomicCapture
    use omp_lib                                                                                                       
    integer :: x, y

!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: omp.atomic.capture   memory_order(release) {
!CHECK: omp.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: omp.atomic.update %[[Y]] : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[ARG]] : i32
!CHECK: omp.yield(%[[result]] : i32)
!CHECK: }
!CHECK: }

    !$omp atomic capture release
        x = y
        y = x + y
    !$omp end atomic


!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: omp.atomic.capture   hint(uncontended) {
!CHECK: omp.atomic.update %[[Y]] : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.muli %[[temp]], %[[ARG]] : i32
!CHECK: omp.yield(%[[result]] : i32)
!CHECK: }
!CHECK: omp.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: }

    !$omp atomic hint(omp_sync_hint_uncontended) capture
        y = x * y 
        x = y
    !$omp end atomic

!CHECK: %[[constant_20:.*]] = arith.constant 20 : i32
!CHECK: %[[constant_8:.*]] = arith.constant 8 : i32
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.subi %[[constant_8]], %[[temp]] : i32
!CHECK: %[[result_noreassoc:.*]] = fir.no_reassoc %[[result]] : i32
!CHECK: %[[result:.*]] = arith.addi %[[constant_20]], %[[result_noreassoc]] : i32
!CHECK: omp.atomic.capture   memory_order(acquire) hint(nonspeculative) {
!CHECK: omp.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: omp.atomic.write %[[Y]] = %[[result]] : !fir.ref<i32>, i32
!CHECK: }

    !$omp atomic hint(omp_lock_hint_nonspeculative) capture acquire
        x = y
        y = 2 * 10 + (8 - x) 
    !$omp end atomic


!CHECK: %[[constant_20:.*]] = arith.constant 20 : i32
!CHECK: %[[constant_8:.*]] = arith.constant 8 : i32
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.subi %[[constant_8]], %[[temp]] : i32
!CHECK: %[[result_noreassoc:.*]] = fir.no_reassoc %[[result]] : i32
!CHECK: %[[result:.*]] = arith.addi %[[constant_20]], %[[result_noreassoc]] : i32
!CHECK: omp.atomic.capture {
!CHECK: omp.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: omp.atomic.write %[[Y]] = %[[result]] : !fir.ref<i32>, i32
!CHECK: }

    !$omp atomic capture
        x = y
        y = 2 * 10 + (8 - x) 
    !$omp end atomic 
end program



subroutine pointers_in_atomic_capture()
!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFpointers_in_atomic_captureEa"}
!CHECK: {{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFpointers_in_atomic_captureEb"}
!CHECK: {{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", fir.target, uniq_name = "_QFpointers_in_atomic_captureEc"}
!CHECK: %[[D:.*]] = fir.alloca i32 {bindc_name = "d", fir.target, uniq_name = "_QFpointers_in_atomic_captureEd"}
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_A_addr:.*]] = fir.box_addr %[[loaded_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_B_addr:.*]] = fir.box_addr %[[loaded_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[PRIVATE_LOADED_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[PRIVATE_LOADED_B_addr:.*]] = fir.box_addr %[[PRIVATE_LOADED_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_value:.*]] = fir.load %[[PRIVATE_LOADED_B_addr]] : !fir.ptr<i32>
!CHECK: omp.atomic.capture   {
!CHECK: omp.atomic.update %[[loaded_A_addr]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[ARG]], %[[loaded_value]] : i32
!CHECK: omp.yield(%[[result]] : i32)
!CHECK: }
!CHECK: omp.atomic.read %[[loaded_B_addr]] = %[[loaded_A_addr]] : !fir.ptr<i32>, i32
!CHECK: }
    integer, pointer :: a, b
    integer, target :: c, d
    a=>c
    b=>d

    !$omp atomic capture
        a = a + b
        b = a
    !$omp end atomic
end subroutine
