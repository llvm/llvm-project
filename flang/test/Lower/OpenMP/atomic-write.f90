! REQUIRES: openmp_runtime

! RUN: bbc %openmp_flags -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

! This test checks the lowering of atomic write

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomicwrite"} {
!CHECK:    %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[Y_REF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK:    %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[Z_REF:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK:    %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z_REF]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C44:.*]] = arith.constant 44 : i32
!CHECK:    omp.atomic.write %[[X_DECL:.*]]#0 = %[[C44]]   hint(uncontended) memory_order(seq_cst) : !fir.ref<i32>, i32
!CHECK:    %[[C7:.*]] = arith.constant 7 : i32
!CHECK:    %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[SEVEN_Y_VAL:.*]] = arith.muli %[[C7]], %[[Y_VAL]] : i32
!CHECK:    omp.atomic.write %[[X_DECL]]#0 = %[[SEVEN_Y_VAL]]   memory_order(relaxed) : !fir.ref<i32>, i32
!CHECK:    %[[C10:.*]] = arith.constant 10 : i32
!CHECK:    %[[X_VAL:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[TEN_X:.*]] = arith.muli %[[C10]], %[[X_VAL]] : i32
!CHECK:    %[[Z_VAL:.*]] = fir.load %[[Z_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[C2:.*]] = arith.constant 2 : i32
!CHECK:    %[[Z_DIV_2:.*]] = arith.divsi %[[Z_VAL]], %[[C2]] : i32
!CHECK:    %[[ADD_RES:.*]] = arith.addi %[[TEN_X]], %[[Z_DIV_2]] : i32
!CHECK:    omp.atomic.write %[[Y_DECL]]#0 = %[[ADD_RES]]   hint(speculative) memory_order(release) : !fir.ref<i32>, i32

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

!CHECK-LABEL: func.func @_QPatomic_write_pointer() {
!CHECK:    %[[X_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_write_pointerEx"}
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_write_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:    %[[X_ADDR_BOX:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[X_POINTEE_ADDR:.*]] = fir.box_addr %[[X_ADDR_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    %[[C1:.*]] = arith.constant 1 : i32
!CHECK:    omp.atomic.write %[[X_POINTEE_ADDR]] = %[[C1]]   : !fir.ptr<i32>, i32
!CHECK:    %[[C2:.*]] = arith.constant 2 : i32
!CHECK:    %[[X_ADDR_BOX:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[X_POINTEE_ADDR:.*]] = fir.box_addr %[[X_ADDR_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    hlfir.assign %[[C2]] to %[[X_POINTEE_ADDR]] : i32, !fir.ptr<i32>

subroutine atomic_write_pointer()
  integer, pointer :: x

  !$omp atomic write
    x = 1

  x = 2
end

!CHECK-LABEL: func.func @_QPatomic_write_typed_assign
!CHECK:    %[[R2_REF:.*]] = fir.alloca f32 {bindc_name = "r2", uniq_name = "_QFatomic_write_typed_assignEr2"}
!CHECK:    %[[R2_DECL:.*]]:2 = hlfir.declare %[[R2_REF]] {uniq_name = "_QFatomic_write_typed_assignEr2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:    %[[C0:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:    omp.atomic.write %[[R2_DECL]]#0 = %[[C0]]   : !fir.ref<f32>, f32

subroutine atomic_write_typed_assign
  real :: r2
  !$omp atomic write
  r2 = 0
end subroutine

!CHECK-LABEL: func.func @_QPatomic_write_logical()
!CHECK:    %[[L_REF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFatomic_write_logicalEl"}
!CHECK:    %[[L_DECL:.*]]:2 = hlfir.declare %[[L_REF]] {uniq_name = "_QFatomic_write_logicalEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:    %true = arith.constant true
!CHECK:    %[[CVT:.*]] = fir.convert %true : (i1) -> !fir.logical<4>
!CHECK:    omp.atomic.write %[[L_DECL]]#0 = %[[CVT]] : !fir.ref<!fir.logical<4>>, !fir.logical<4>

subroutine atomic_write_logical
  logical :: l
  !$omp atomic write
      l = .true.
  !$omp end atomic
end
