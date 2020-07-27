! This test checks lowering of OpenMP taskwait Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program taskwait

        integer :: a,b,c

!$OMP TASKWAIT
!FIRDialect: omp.taskwait
!LLVMIRDialect: omp.taskwait
!LLVMIR: %{{.*}} = call i32 @__kmpc_omp_taskwait(%struct.ident_t* @{{.*}}, i32 %{{.*}})
        c = a + b
!$OMP TASKWAIT
!FIRDialect: omp.taskwait
!LLVMIRDialect: omp.taskwait
!LLVMIR: %{{.*}} = call i32 @__kmpc_omp_taskwait(%struct.ident_t* @{{.*}}, i32 %{{.*}})

end program
