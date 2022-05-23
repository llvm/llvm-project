! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! This test checks the lowering of atomic write

!CHECK: func @_QQmain() {
!CHECK: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[VAR_Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[CONST_44:.*]] = arith.constant 44 : i32
!CHECK: omp.atomic.write %[[VAR_X]] = %[[CONST_44]] hint(uncontended) memory_order(seq_cst) : !fir.ref<i32>, i32
!CHECK: %[[CONST_7:.*]] = arith.constant 7 : i32
!CHECK: {{.*}} = fir.load %[[VAR_Y]] : !fir.ref<i32>
!CHECK: %[[VAR_7y:.*]] = arith.muli %[[CONST_7]], {{.*}} : i32
!CHECK: omp.atomic.write %[[VAR_X]] = %[[VAR_7y]] memory_order(relaxed) : !fir.ref<i32>, i32
!CHECK: %[[CONST_10:.*]] = arith.constant 10 : i32
!CHECK: {{.*}} = fir.load %[[VAR_X]] : !fir.ref<i32>
!CHECK: {{.*}} = arith.muli %[[CONST_10]], {{.*}} : i32
!CHECK: {{.*}} = fir.load %[[VAR_Z]] : !fir.ref<i32>
!CHECK: %[[CONST_2:.*]] = arith.constant 2 : i32
!CHECK: {{.*}} = arith.divsi {{.*}}, %[[CONST_2]] : i32
!CHECK: {{.*}} = arith.addi {{.*}}, {{.*}} : i32
!CHECK: omp.atomic.write %[[VAR_Y]] = {{.*}} hint(speculative) memory_order(release) : !fir.ref<i32>, i32
!CHECK: return
!CHECK: }

program OmpAtomicWrite
    use omp_lib
    integer :: x, y, z
    !$omp atomic seq_cst write hint(omp_sync_hint_uncontended)
        x = 8*4 + 12

    !$omp atomic write relaxed
        x = 7 * y

    !$omp atomic write release hint(omp_sync_hint_speculative)
        y = 10*x + z/2
end program OmpAtomicWrite

