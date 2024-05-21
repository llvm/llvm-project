! REQUIRES: openmp_runtime

! RUN: bbc %openmp_flags -emit-hlfir %s -o - | FileCheck %s

! This test checks the lowering of atomic read

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomic"} {
!CHECK:    %[[A_C1:.*]] = arith.constant 1 : index
!CHECK:    %[[A_REF:.*]] = fir.alloca !fir.char<1> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK:    %[[A_DECL:.*]]:2 = hlfir.declare %[[A_REF]] typeparams %[[A_C1]] {uniq_name = "_QFEa"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK:    %[[B_C1:.*]] = arith.constant 1 : index
!CHECK:    %[[B_REF:.*]] = fir.alloca !fir.char<1> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK:    %[[B_DECL:.*]]:2 = hlfir.declare %[[B_REF]] typeparams %[[B_C1]] {uniq_name = "_QFEb"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK:    %[[C_REF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "c", uniq_name = "_QFEc"}
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_REF]] {uniq_name = "_QFEc"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:    %[[D_REF:.*]] = fir.alloca !fir.logical<4> {bindc_name = "d", uniq_name = "_QFEd"}
!CHECK:    %[[D_DECL:.*]]:2 = hlfir.declare %[[D_REF]] {uniq_name = "_QFEd"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:    %[[E_C8:.*]] = arith.constant 8 : index
!CHECK:    %[[E_REF:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "e", uniq_name = "_QFEe"}
!CHECK:    %[[E_DECL:.*]]:2 = hlfir.declare %[[E_REF]] typeparams %[[E_C8]] {uniq_name = "_QFEe"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
!CHECK:    %[[F_C8:.*]] = arith.constant 8 : index
!CHECK:    %[[F_REF:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "f", uniq_name = "_QFEf"}
!CHECK:    %[[F_DECL:.*]]:2 = hlfir.declare %[[F_REF]] typeparams %[[F_C8]] {uniq_name = "_QFEf"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
!CHECK:    %[[G_REF:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!CHECK:    %[[G_DECL:.*]]:2 = hlfir.declare %[[G_REF]] {uniq_name = "_QFEg"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:    %[[H_REF:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!CHECK:    %[[H_DECL:.*]]:2 = hlfir.declare %[[H_REF]] {uniq_name = "_QFEh"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:    %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[Y_REF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK:    %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.atomic.read %[[X_DECL]]#1 = %[[Y_DECL]]#1   memory_order(acquire) hint(uncontended) : !fir.ref<i32>, i32
!CHECK:    omp.atomic.read %[[A_DECL]]#1 = %[[B_DECL]]#1   memory_order(relaxed) : !fir.ref<!fir.char<1>>, !fir.char<1>
!CHECK:    omp.atomic.read %[[C_DECL]]#1 = %[[D_DECL]]#1   memory_order(seq_cst) hint(contended) : !fir.ref<!fir.logical<4>>, !fir.logical<4>
!CHECK:    omp.atomic.read %[[E_DECL]]#1 = %[[F_DECL]]#1   hint(speculative) : !fir.ref<!fir.char<1,8>>, !fir.char<1,8>
!CHECK:    omp.atomic.read %[[G_DECL]]#1 = %[[H_DECL]]#1   hint(nonspeculative) : !fir.ref<f32>, f32
!CHECK:    omp.atomic.read %[[G_DECL]]#1 = %[[H_DECL]]#1   : !fir.ref<f32>, f32

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
!CHECK:    %[[X_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
!CHECK:    fir.store %2 to %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:    %[[Y_REF:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
!CHECK:    %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEy"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:    %[[X_ADDR:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[X_POINTEE_ADDR:.*]] = fir.box_addr %[[X_ADDR]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    %[[Y_ADDR:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[Y_POINTEE_ADDR:.*]] = fir.box_addr %[[Y_ADDR]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    omp.atomic.read %[[Y_POINTEE_ADDR]] = %[[X_POINTEE_ADDR]]   : !fir.ptr<i32>, i32
!CHECK:    %[[Y_ADDR:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[Y_POINTEE_ADDR:.*]] = fir.box_addr %[[Y_ADDR]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    %[[Y_POINTEE_VAL:.*]] = fir.load %[[Y_POINTEE_ADDR]] : !fir.ptr<i32>
!CHECK:    %[[X_ADDR:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[X_POINTEE_ADDR:.*]] = fir.box_addr %[[X_ADDR]] : (!fir.box<!fir.ptr<i32>>
!CHECK:    hlfir.assign %[[Y_POINTEE_VAL]] to %[[X_POINTEE_ADDR]] : i32, !fir.ptr<i32>

subroutine atomic_read_pointer()
  integer, pointer :: x, y

  !$omp atomic read
    y = x

  x = y
end

