! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! This test checks the lowering of atomic read

!CHECK: func @_QQmain() {
!CHECK: %[[VAR_A:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.char<1>>
!CHECK: %[[VAR_B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.char<1>>
!CHECK: %[[VAR_C:.*]] = fir.alloca !fir.logical<4> {bindc_name = "c", uniq_name = "_QFEc"}
!CHECK: %[[VAR_D:.*]] = fir.alloca !fir.logical<4> {bindc_name = "d", uniq_name = "_QFEd"}
!CHECK: %[[VAR_E:.*]] = fir.address_of(@_QFEe) : !fir.ref<!fir.char<1,8>>
!CHECK: %[[VAR_F:.*]] = fir.address_of(@_QFEf) : !fir.ref<!fir.char<1,8>>
!CHECK: %[[VAR_G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!CHECK: %[[VAR_H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!CHECK: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: omp.atomic.read %[[VAR_X]] = %[[VAR_Y]] memory_order(acquire)  hint(uncontended) : !fir.ref<i32>
!CHECK: omp.atomic.read %[[VAR_A]] = %[[VAR_B]] memory_order(relaxed) hint(none)  : !fir.ref<!fir.char<1>>
!CHECK: omp.atomic.read %[[VAR_C]] = %[[VAR_D]] memory_order(seq_cst)  hint(contended) : !fir.ref<!fir.logical<4>>
!CHECK: omp.atomic.read %[[VAR_E]] = %[[VAR_F]] hint(speculative) : !fir.ref<!fir.char<1,8>>
!CHECK: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] hint(nonspeculative) : !fir.ref<f32>
!CHECK: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] : !fir.ref<f32>
!CHECK: return
!CHECK: }

program OmpAtomic

    use omp_lib
    integer :: x, y
    character :: a, b
    logical :: c, d
    character(8) :: e, f
    real g, h
    !$omp atomic acquire read hint(omp_sync_hint_uncontended)
       x = y
    !$omp atomic relaxed read hint(omp_sync_hint_none)
       a = b
    !$omp atomic read seq_cst hint(omp_sync_hint_contended)
       c = d
    !$omp atomic read hint(omp_sync_hint_speculative)
       e = f
    !$omp atomic read hint(omp_sync_hint_nonspeculative)
       g = h
    !$omp atomic read
       g = h
end program OmpAtomic

