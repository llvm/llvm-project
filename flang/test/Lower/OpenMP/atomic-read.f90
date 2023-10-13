! This test checks the lowering of atomic read
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-hlfir %s -o - | FileCheck %s

!CHECK: func @_QQmain() attributes {fir.bindc_name = "ompatomic"} {
!CHECK: %[[CONST_1:.*]] = arith.constant 1 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.char<1> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] typeparams %[[CONST_1]] {uniq_name = "_QFEa"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK: %[[CONST_1_0:.*]] = arith.constant 1 : index
!CHECK: %[[B:.*]] = fir.alloca !fir.char<1> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] typeparams %[[CONST_1_0]] {uniq_name = "_QFEb"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK: %[[C:.*]] = fir.alloca !fir.logical<4> {bindc_name = "c", uniq_name = "_QFEc"}
!CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] {uniq_name = "_QFEc"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK: %[[D:.*]] = fir.alloca !fir.logical<4> {bindc_name = "d", uniq_name = "_QFEd"}
!CHECK: %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QFEd"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK: %[[CONST_8:.*]] = arith.constant 8 : index
!CHECK: %[[E:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "e", uniq_name = "_QFEe"}
!CHECK: %[[E_DECL:.*]]:2 = hlfir.declare %[[E]] typeparams %[[CONST_8]] {uniq_name = "_QFEe"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
!CHECK: %[[CONST_8_1:.*]] = arith.constant 8 : index
!CHECK: %[[F:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "f", uniq_name = "_QFEf"}
!CHECK: %[[F_DECL:.*]]:2 = hlfir.declare %[[F]] typeparams %[[CONST_8_1]] {uniq_name = "_QFEf"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
!CHECK: %[[G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!CHECK: %[[G_DECL:.*]]:2 = hlfir.declare %[[G]] {uniq_name = "_QFEg"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!CHECK: %[[H_DECL:.*]]:2 = hlfir.declare %[[H]] {uniq_name = "_QFEh"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.atomic.read %[[X_DECL]]#1 = %[[Y_DECL]]#1   memory_order(acquire)  hint(uncontended) : !fir.ref<i32>, i32
!CHECK: omp.atomic.read %[[A_DECL]]#1 = %[[B_DECL]]#1   memory_order(relaxed) : !fir.ref<!fir.char<1>>, !fir.char<1>
!CHECK: omp.atomic.read %[[C_DECL]]#1 = %[[D_DECL]]#1   memory_order(seq_cst)  hint(contended) : !fir.ref<!fir.logical<4>>, !fir.logical<4>
!CHECK: omp.atomic.read %[[E_DECL]]#1 = %[[F_DECL]]#1   hint(speculative) : !fir.ref<!fir.char<1,8>>, !fir.char<1,8>
!CHECK: omp.atomic.read %[[G_DECL]]#1 = %[[H_DECL]]#1   hint(nonspeculative) : !fir.ref<f32>, f32
!CHECK: omp.atomic.read %[[G_DECL]]#1 = %[[H_DECL]]#1   : !fir.ref<f32>, f32
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
! Please notice to use %[[VAL_10]] and %[[VAL_8]] for operands of atomic
! operation, instead of %[[VAL_3]] and %[[VAL_7]].

!CHECK-LABEL: func.func @_QPatomic_read_pointer() {
!CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
!CHECK:         %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
!CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEy"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         omp.atomic.read %[[VAL_11:.*]] = %[[VAL_9]]   : !fir.ptr<i32>, i32
!CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ptr<i32>
!CHECK:         %[[VAL_15:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:         %[[VAL_16:.*]] = fir.box_addr %[[VAL_15]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:         hlfir.assign %[[VAL_14]] to %[[VAL_16]] : i32, !fir.ptr<i32>
!CHECK:         return
!CHECK:       }

subroutine atomic_read_pointer()
 integer, pointer :: x, y

 !$omp atomic read
   y = x

 x = y
end

