! This test checks the lowering of OpenMP sections construct with several clauses present

! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix="FIRDialect"
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefix="LLVMDialect"
! TODO before (%flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | tco | FileCheck %s --check-prefix="LLVMIR"):
! ensure allocate clause lowering

!FIRDialect: func @_QQmain() {
!FIRDialect: %[[COUNT:.*]] = fir.address_of(@_QFEcount) : !fir.ref<i32> 
!FIRDialect: %[[DOUBLE_COUNT:.*]] = fir.address_of(@_QFEdouble_count) : !fir.ref<i32>
!FIRDialect: %[[ETA:.*]] = fir.alloca f32 {bindc_name = "eta", uniq_name = "_QFEeta"}
!FIRDialect: %[[CONST_1:.*]] = arith.constant 1 : i32
!FIRDialect: omp.sections allocate(%[[CONST_1]] : i32 -> %0 : !fir.ref<i32>)  {
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = arith.constant 5 : i32
!FIRDialect: fir.store {{.*}} to {{.*}} : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = arith.muli {{.*}}, {{.*}} : i32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: fir.store {{.*}} to %[[ETA]] : !fir.ref<f32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = fir.load %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = arith.constant 1 : i32
!FIRDialect: {{.*}} = arith.addi {{.*}} : i32
!FIRDialect: fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.section {
!FIRDialect: {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.constant 7.000000e+00 : f32
!FIRDialect: {{.*}} = arith.subf {{.*}} : f32
!FIRDialect: fir.store {{.*}} to %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.mulf {{.*}}, {{.*}} : f32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (f32) -> i32
!FIRDialect: fir.store {{.*}} to %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!FIRDialect: {{.*}} = fir.convert {{.*}} : (i32) -> f32
!FIRDialect: {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!FIRDialect: {{.*}} = arith.subf {{.*}}, {{.*}} : f32
!FIRDialect: {{.*}} = fir.convert {{.*}} : (f32) -> i32
!FIRDialect: fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.sections nowait {
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: return
!FIRDialect: }

!LLVMDialect: llvm.func @_QQmain() {
!LLVMDialect: %[[COUNT:.*]] = llvm.mlir.addressof @_QFEcount : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = builtin.unrealized_conversion_cast %[[COUNT]] : !llvm.ptr<i32> to !fir.ref<i32>
!LLVMDialect: %[[DOUBLE_COUNT:.*]] = llvm.mlir.addressof @_QFEdouble_count : !llvm.ptr<i32>
!LLVMDialect: %[[ALLOCATOR:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMDialect: %[[ETA:.*]] = llvm.alloca %[[ALLOCATOR]] x f32 {bindc_name = "eta", in_type = f32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFEeta"} : (i64) -> !llvm.ptr<f32>
!LLVMDialect: %[[CONSTANT:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMDialect: omp.sections   allocate(%[[CONSTANT]] : i32 -> %1 : !fir.ref<i32>) {
!LLVMDialect: omp.section {
!LLVMDialect: {{.*}} = llvm.mlir.constant(5 : i32) : i32
!LLVMDialect: llvm.store {{.*}}, %[[COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.load %[[COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.load %[[DOUBLE_COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.mul {{.*}}, {{.*}} : i32
!LLVMDialect: {{.*}} = llvm.sitofp {{.*}} : i32 to f32
!LLVMDialect: llvm.store {{.*}}, %[[ETA]] : !llvm.ptr<f32>
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.section {
!LLVMDialect: {{.*}} = llvm.load %[[DOUBLE_COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(1 : i32) : i32
!LLVMDialect: {{.*}} = llvm.add {{.*}}, {{.*}} : i32
!LLVMDialect: llvm.store {{.*}}, %[[DOUBLE_COUNT]] : !llvm.ptr<i32>
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.section {
!LLVMDialect: {{.*}} = llvm.load %[[ETA]] : !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(7.000000e+00 : f32) : f32
!LLVMDialect: {{.*}} = llvm.fsub {{.*}}, {{.*}} : f32
!LLVMDialect: llvm.store {{.*}}, %[[ETA]] : !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.load %[[COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.sitofp {{.*}} : i32 to f32
!LLVMDialect: {{.*}} = llvm.load %[[ETA]] : !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.fmul {{.*}}, {{.*}} : f32
!LLVMDialect: {{.*}} = llvm.fptosi {{.*}} : f32 to i32
!LLVMDialect: llvm.store {{.*}}, %[[COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.load %[[COUNT]] : !llvm.ptr<i32>
!LLVMDialect: {{.*}} = llvm.sitofp {{.*}} : i32 to f32
!LLVMDialect: {{.*}} = llvm.load %[[ETA]] : !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.fsub {{.*}}, {{.*}} : f32
!LLVMDialect: {{.*}} = llvm.fptosi {{.*}} : f32 to i32
!LLVMDialect: llvm.store {{.*}}, %[[DOUBLE_COUNT]] : !llvm.ptr<i32>
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.sections nowait {
!LLVMDialect: omp.section {
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: llvm.return
!LLVMDialect: }

program sample 
    use omp_lib
    integer :: count = 0, double_count = 1
    !$omp sections private (eta, double_count) allocate(omp_high_bw_mem_alloc: count)
        !$omp section
            count = 1 + 4
            eta = count * double_count
        !$omp section
            double_count = double_count + 1
        !$omp section
            eta = eta - 7
            count = count * eta
            double_count = count - eta
    !$omp end sections

    !$omp sections
    !$omp end sections nowait
end program sample

!FIRDialect: func @_QPfirstprivate(%[[ARG:.*]]: !fir.ref<f32> {fir.bindc_name = "alpha"}) {
!FIRDialect: omp.sections {
!FIRDialect: omp.section  {
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.sections {
!FIRDialect: omp.section  {
!FIRDialect: %[[PRIVATE_VAR:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!FIRDialect: %[[CONSTANT:.*]] = arith.constant 5.000000e+00 : f32
!FIRDialect: %[[PRIVATE_VAR_2:.*]] = arith.mulf %[[PRIVATE_VAR]], %[[CONSTANT]] : f32
!FIRDialect: fir.store %[[PRIVATE_VAR_2]] to %[[ARG]] : !fir.ref<f32>
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: omp.terminator
!FIRDialect: }
!FIRDialect: return
!FIRDialect: }

!LLVMDialect: llvm.func @_QPfirstprivate(%[[ARG:.*]]: !llvm.ptr<f32> {fir.bindc_name = "alpha"}) {
!LLVMDialect: omp.sections   {
!LLVMDialect: omp.section {
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.sections {
!LLVMDialect: omp.section {
!LLVMDialect: {{.*}} = llvm.load %[[ARG]] : !llvm.ptr<f32>
!LLVMDialect: {{.*}} = llvm.mlir.constant(5.000000e+00 : f32) : f32
!LLVMDialect: {{.*}} = llvm.fmul {{.*}}, {{.*}} : f32
!LLVMDialect: llvm.store {{.*}}, %[[ARG]] : !llvm.ptr<f32>
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: omp.terminator
!LLVMDialect: }
!LLVMDialect: llvm.return
!LLVMDialect: }

subroutine firstprivate(alpha)
    real :: alpha 
    !$omp sections firstprivate(alpha)
    !$omp end sections

    !$omp sections
        alpha = alpha * 5
    !$omp end sections
end subroutine
