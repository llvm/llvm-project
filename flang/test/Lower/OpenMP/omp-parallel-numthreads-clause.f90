! This test checks lowering of OpenMP parallel Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program parallel

        integer :: a,b,c
        integer :: num_threads
! This and last statements are just for the sake ensuring that the
! operation is created/inserted correctly and does not break/interfere with
! other pieces which may be present before/after the operation.
! However this test does not verify operation corresponding to this
! statment.
        c = a + b
!$OMP PARALLEL
!$OMP END PARALLEL
!FIRDialect: omp.parallel {
!FIRDialect-NEXT: omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect: omp.parallel {
!LLVMIRDialect-NEXT: omp.terminator
!LLVMIRDialect-NEXT: }

!$OMP PARALLEL NUM_THREADS(16)
!$OMP END PARALLEL
        num_threads = 4
!$OMP PARALLEL NUM_THREADS(num_threads)
!$OMP END PARALLEL

!FIRDialect: omp.parallel num_threads(%{{.*}} : i32) {
!FIRDialect-NEXT: omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect: omp.parallel num_threads(%{{.*}} : !llvm.i32) {
!LLVMIRDialect-NEXT: omp.terminator
!LLVMIRDialect-NEXT: }


!LLVMIR-LABEL: call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.*}})
!LLVMIR: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN:.*]] to {{.*}}

!LLVMIR: %[[GLOBAL_THREAD_NUM1:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.*}})
!LLVMIR: call void @__kmpc_push_num_threads(%struct.ident_t* @{{.*}}, i32 %[[GLOBAL_THREAD_NUM1]], i32 16)
!LLVMIR: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN1:.*]] to {{.*}}

!LLVMIR: %[[GLOBAL_THREAD_NUM2:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.*}})
!LLVMIR: call void @__kmpc_push_num_threads(%struct.ident_t* @{{.*}}, i32 %[[GLOBAL_THREAD_NUM2]], i32 %{{.*}})
!LLVMIR: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN2:.*]] to {{.*}}

!LLVMIR: define internal void @[[OMP_OUTLINED_FN2]]
!LLVMIR: define internal void @[[OMP_OUTLINED_FN1]]
!LLVMIR: define internal void @[[OMP_OUTLINED_FN]]
        b = a + c

end program
