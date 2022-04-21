! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN: FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | \
! RUN: FileCheck %s --check-prefix=LLVMDialect

! This test checks the lowering of atomic read

!FIRDialect: func @_QQmain() {
!FIRDialect: %[[VAR_A:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.char<1>>
!FIRDialect: %[[VAR_B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.char<1>>
!FIRDialect: %[[VAR_C:.*]] = fir.alloca !fir.logical<4> {bindc_name = "c", uniq_name = "_QFEc"}
!FIRDialect: %[[VAR_D:.*]] = fir.alloca !fir.logical<4> {bindc_name = "d", uniq_name = "_QFEd"}
!FIRDialect: %[[VAR_E:.*]] = fir.address_of(@_QFEe) : !fir.ref<!fir.char<1,8>>
!FIRDialect: %[[VAR_F:.*]] = fir.address_of(@_QFEf) : !fir.ref<!fir.char<1,8>>
!FIRDialect: %[[VAR_G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!FIRDialect: %[[VAR_H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!FIRDialect: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!FIRDialect: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!FIRDialect: omp.atomic.read %[[VAR_X]] = %[[VAR_Y]] memory_order(acquire)  hint(uncontended) : !fir.ref<i32>
!FIRDialect: omp.atomic.read %[[VAR_A]] = %[[VAR_B]] memory_order(relaxed) hint(none)  : !fir.ref<!fir.char<1>>
!FIRDialect: omp.atomic.read %[[VAR_C]] = %[[VAR_D]] memory_order(seq_cst)  hint(contended) : !fir.ref<!fir.logical<4>>
!FIRDialect: omp.atomic.read %[[VAR_E]] = %[[VAR_F]] hint(speculative) : !fir.ref<!fir.char<1,8>>
!FIRDialect: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] hint(nonspeculative) : !fir.ref<f32>
!FIRDialect: omp.atomic.read %[[VAR_G]] = %[[VAR_H]] : !fir.ref<f32>
!FIRDialect: return
!FIRDialect: }

!LLVMDialect: llvm.func @_QQmain() {
!LLVMDialect: %[[LLVM_VAR_A:.*]] = llvm.mlir.addressof @_QFEa : !llvm.ptr<array<1 x i8>>
!LLVMDialect: %[[LLVM_VAR_B:.*]] = llvm.mlir.addressof @_QFEb : !llvm.ptr<array<1 x i8>>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect:  %[[LLVM_VAR_C:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "c", in_type = !fir.logical<4>, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEc"} : (i64) -> !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[LLVM_VAR_D:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "d", in_type = !fir.logical<4>, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEd"} : (i64) -> !llvm.ptr<i32>
!LLVMDialect: %[[LLVM_VAR_E:.*]] = llvm.mlir.addressof @_QFEe : !llvm.ptr<array<8 x i8>>
!LLVMDialect: %[[LLVM_VAR_F:.*]] = llvm.mlir.addressof @_QFEf : !llvm.ptr<array<8 x i8>>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[LLVM_VAR_G:.*]] = llvm.alloca {{.*}} x f32 {bindc_name = "g", in_type = f32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEg"} : (i64) -> !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[LLVM_VAR_H:.*]] = llvm.alloca {{.*}} x f32 {bindc_name = "h", in_type = f32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEh"} : (i64) -> !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[LLVM_VAR_X:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "x", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEx"} : (i64) -> !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[LLVM_VAR_Y:.*]] = llvm.alloca {{.*}} x i32 {bindc_name = "y", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEy"} : (i64) -> !llvm.ptr<i32>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_X]] = %[[LLVM_VAR_Y]] memory_order(acquire)  hint(uncontended) : !llvm.ptr<i32>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_A]] =  %[[LLVM_VAR_B]] memory_order(relaxed) hint(none) : !llvm.ptr<array<1 x i8>>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_C]] = %[[LLVM_VAR_D]] memory_order(seq_cst)  hint(contended) : !llvm.ptr<i32>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_E]] = %[[LLVM_VAR_F]] hint(speculative) : !llvm.ptr<array<8 x i8>>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_G]] = %[[LLVM_VAR_H]] hint(nonspeculative) : !llvm.ptr<f32>
!LLVMDialect: omp.atomic.read %[[LLVM_VAR_G]] = %[[LLVM_VAR_H]] : !llvm.ptr<f32>
!LLVMDialect: llvm.return
!LLVMDialect: }

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
