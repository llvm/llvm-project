! This test checks the lowering of atomic write
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-hlfir %s -o - | FileCheck %s

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomicwrite"} {
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST_44:.*]] = arith.constant 44 : i32
!CHECK: omp.atomic.write %[[X_DECL]]#1 = %[[CONST_44]] hint(uncontended) memory_order(seq_cst) : !fir.ref<i32>, i32
!CHECK: %[[CONST_7:.*]] = arith.constant 7 : i32
!CHECK: {{.*}} = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[VAR_7y:.*]] = arith.muli %[[CONST_7]], {{.*}} : i32
!CHECK: omp.atomic.write %[[X_DECL]]#1 = %[[VAR_7y]]   memory_order(relaxed) : !fir.ref<i32>, i32
!CHECK: %[[CONST_10:.*]] = arith.constant 10 : i32
!CHECK: {{.*}} = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: {{.*}} = arith.muli %[[CONST_10]], {{.*}} : i32
!CHECK: {{.*}} = fir.load %[[Z_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[CONST_2:.*]] = arith.constant 2 : i32
!CHECK: {{.*}} = arith.divsi {{.*}}, %[[CONST_2]] : i32
!CHECK: {{.*}} = arith.addi {{.*}}, {{.*}} : i32
!CHECK: omp.atomic.write %[[Y_DECL]]#1 = {{.*}} hint(speculative) memory_order(release) : !fir.ref<i32>, i32
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
! Please notice to use %[[VAL_4]] for operands of atomic operation, instead
! of %[[VAL_3]].

!CHECK-LABEL: func.func @_QPatomic_write_pointer() {
!CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_write_pointerEx"}
!CHECK:         %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_write_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:         %[[CONST_1:.*]] = arith.constant 1 : i32
!CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         omp.atomic.write %[[VAL_5:.*]] = %[[CONST_1]]   : !fir.ptr<i32>, i32
!CHECK:         %[[CONST_2:.*]] = arith.constant 2 : i32
!CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         hlfir.assign %[[CONST_2:.*]] to %[[VAL_7]] : i32, !fir.ptr<i32>
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
!CHECK: %[[VAR_DECL:.*]]:2 = hlfir.declare %[[VAR]] {uniq_name = "{{.*}}r2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: omp.atomic.write %[[VAR_DECL]]#1 = %[[CST]]   : !fir.ref<f32>, f32

subroutine atomic_write_typed_assign
  real :: r2
  !$omp atomic write
  r2 = 0
end subroutine
