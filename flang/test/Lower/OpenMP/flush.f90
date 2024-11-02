! This test checks lowering of OpenMP Flush Directive.

!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --cfg-conversion | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="LLVMIRDialect,OMPDialect"

subroutine flush_standalone(a, b, c)
    integer, intent(inout) :: a, b, c

!$omp flush(a,b,c)
!$omp flush
!OMPDialect: omp.flush(%{{.*}}, %{{.*}}, %{{.*}} :
!FIRDialect: !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!LLVMIRDialect: !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>)
!OMPDialect: omp.flush

end subroutine flush_standalone

subroutine flush_parallel(a, b, c)
    integer, intent(inout) :: a, b, c

!$omp parallel
!OMPDialect:  omp.parallel {

!OMPDialect: omp.flush(%{{.*}}, %{{.*}}, %{{.*}} :
!FIRDialect: !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!LLVMIRDialect: !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>)
!OMPDialect: omp.flush
!$omp flush(a,b,c)
!$omp flush

!FIRDialect: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
!FIRDialect: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>

!LLVMIRDialect: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect: %{{.*}} = llvm.add %{{.*}}, %{{.*}} : i32
!LLVMIRDialect: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<i32>
    c = a + b

!OMPDialect: omp.terminator
!$omp END parallel

end subroutine flush_parallel
