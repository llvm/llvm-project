! This test checks lowering of OpenMP parallel Directive with a loop inside.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR


program main
!$OMP PARALLEL
      do i = 2, 5
      end do
!$OMP END PARALLEL
end
!FIRDialect: omp.parallel {
!FIRDialect: omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect: omp.parallel {
!LLVMIRDialect:   ^[[BB1:.*]]({{.*}}):
!LLVMIRDialect:     %[[COND:.*]] = llvm.icmp "sgt" {{.*}}
!LLVMIRDialect:     llvm.cond_br %[[COND]], ^[[BB2:.*]], ^[[BB3:.*]]
!LLVMIRDialect:   ^[[BB2]]:
!LLVMIRDialect:     llvm.br ^[[BB1]]({{.*}})
!LLVMIRDialect:   ^[[BB3]]:
!LLVMIRDialect:   omp.terminator
!LLVMIRDialect: }

!LLVMIR: call {{.*}} @__kmpc_fork_call(%struct.ident_t* @{{.*}} @_QQmain..omp_par
!LLVMIR: define internal void @_QQmain..omp_par{{.*}} {
!LLVMIR: [[BB1:.*]]: ; preds = %{{.*}}, %{{.*}}
!LLVMIR:   %[[COND:.*]] = icmp sgt i64 {{.*}}
!LLVMIR-NEXT:   br i1 %[[COND]], label %[[BB2:.*]], label %[[BB3:.*]],
!LLVMIR: [[BB3]]:
!LLVMIR: [[BB2]]:
!LLVMIR:   br label %[[BB1]]
!LLVMIR: }
