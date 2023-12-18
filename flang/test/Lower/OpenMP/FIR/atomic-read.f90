! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-fir -hlfir=false %s -o - | FileCheck %s

! This test checks the lowering of atomic read

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomic"} {
!CHECK: %[[VAR_A:.*]] = fir.alloca !fir.char<1> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[VAR_B:.*]] = fir.alloca !fir.char<1> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[VAR_C:.*]] = fir.alloca !fir.logical<4> {bindc_name = "c", uniq_name = "_QFEc"}
!CHECK: %[[VAR_D:.*]] = fir.alloca !fir.logical<4> {bindc_name = "d", uniq_name = "_QFEd"}
!CHECK: %[[VAR_E:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "e", uniq_name = "_QFEe"}
!CHECK: %[[VAR_F:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "f", uniq_name = "_QFEf"}
!CHECK: %[[VAR_G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!CHECK: %[[VAR_H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!CHECK: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: omp.atomic.read %[[VAR_X]] = %[[VAR_Y]] memory_order(acquire)  hint(uncontended) : !fir.ref<i32>, i32
!CHECK: omp.atomic.read %[[VAR_A]] = %[[VAR_B]] memory_order(relaxed) : !fir.ref<!fir.char<1>>, !fir.char<1>
!CHECK: omp.atomic.read %[[VAR_C]] = %[[VAR_D]] memory_order(seq_cst)  hint(contended) : !fir.ref<!fir.logical<4>>, !fir.logical<4>
!CHECK: omp.atomic.read %[[VAR_E]] = %[[VAR_F]] hint(speculative) : !fir.ref<!fir.char<1,8>>, !fir.char<1,8>
!CHECK: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] hint(nonspeculative) : !fir.ref<f32>, f32
!CHECK: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] : !fir.ref<f32>, f32
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

! Test lowering atomic read for pointer variables.
! Please notice to use %[[VAL_4]] and %[[VAL_1]] for operands of atomic
! operation, instead of %[[VAL_3]] and %[[VAL_0]].

!CHECK-LABEL: func.func @_QPatomic_read_pointer() {
!CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
!CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFatomic_read_pointerEx.addr"}
!CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
!CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFatomic_read_pointerEy.addr"}
!CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         omp.atomic.read %[[VAL_7]] = %[[VAL_6]]   : !fir.ptr<i32>, i32
!CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
!CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_9]] to %[[VAL_10]] : !fir.ptr<i32>
!CHECK:         return
!CHECK:       }

subroutine atomic_read_pointer()
  integer, pointer :: x, y

  !$omp atomic read
    y = x

  x = y
end

