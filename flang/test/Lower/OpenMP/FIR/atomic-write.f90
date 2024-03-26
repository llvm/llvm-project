! REQUIRES: openmp_runtime

! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-fir -hlfir=false %s -o - | FileCheck %s

! This test checks the lowering of atomic write

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomicwrite"} {
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

! Test lowering atomic read for pointer variables.
! Please notice to use %[[VAL_1]] for operands of atomic operation, instead
! of %[[VAL_0]].

!CHECK-LABEL: func.func @_QPatomic_write_pointer() {
!CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_write_pointerEx"}
!CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFatomic_write_pointerEx.addr"}
!CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
!CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         omp.atomic.write %[[VAL_4]] = %[[VAL_3]]   : !fir.ptr<i32>, i32
!CHECK:         %[[VAL_5:.*]] = arith.constant 2 : i32
!CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_5]] to %[[VAL_6]] : !fir.ptr<i32>
!CHECK:         return
!CHECK:       }

subroutine atomic_write_pointer()
  integer, pointer :: x

  !$omp atomic write
    x = 1

  x = 2
end

!CHECK-LABEL: func.func @_QPatomic_write_typed_assign
!CHECK: %[[VAR:.*]] = fir.alloca f32 {bindc_name = "r2", uniq_name = "{{.*}}r2"}
!CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: omp.atomic.write %[[VAR]] = %[[CST]]   : !fir.ref<f32>, f32

subroutine atomic_write_typed_assign
  real :: r2
  !$omp atomic write
  r2 = 0
end subroutine
