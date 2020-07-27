! This test checks lowering of OpenMP taskyield Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program taskyield

        integer :: a,b,c

!$OMP TASKYIELD
!FIRDialect: omp.taskyield
!LLVMIRDialect: omp.taskyield
!LLVMIR: %{{.*}} = call i32 @__kmpc_omp_taskyield(%struct.ident_t* @{{.*}}, i32 %{{.*}})
        c = a + b
!$OMP TASKYIELD
!FIRDialect: omp.taskyield
!LLVMIRDialect: omp.taskyield
!LLVMIR: %{{.*}} = call i32 @__kmpc_omp_taskyield(%struct.ident_t* @{{.*}}, i32 %{{.*}})

end program
