! This test checks lowering of OpenMP Barrier Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp %s -o - | \
! RUN:   tco --disable-llvm --print-ir-after=fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program barrier

        integer :: a,b,c

!$OMP BARRIER
!FIRDialect: omp.barrier
!LLVMIRDialect: omp.barrier
!LLVMIR: call void @__kmpc_barrier(%struct.ident_t* @1, i32 %omp_global_thread_num)
        c = a + b
!$OMP BARRIER
!FIRDialect: omp.barrier
!LLVMIRDialect: omp.barrier
!LLVMIR: call void @__kmpc_barrier(%struct.ident_t* @1, i32 %omp_global_thread_num1)

end program
