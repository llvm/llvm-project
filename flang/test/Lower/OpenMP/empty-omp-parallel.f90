! This test checks lowering of OpenMP parallel Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program parallel

        integer :: a,b,c
! This and last statements are just for the sake ensuring that the
! operation is created/inserted correctly and does not break/interfere with
! other pieces which may be present before/after the operation.
! However this test does not verify operation corresponding to this
! statment.
        c = a + b
!$OMP PARALLEL
!FIRDialect: omp.parallel {
!FIRDialect-NEXT: omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect: omp.parallel {
!LLVMIRDialect-NEXT: omp.terminator
!LLVMIRDialect-NEXT: }

!LLVMIR: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN:.*]] to {{.*}}
!LLVMIR: define internal void @[[OMP_OUTLINED_FN]]
!$OMP END PARALLEL
        b = a + c

end program
