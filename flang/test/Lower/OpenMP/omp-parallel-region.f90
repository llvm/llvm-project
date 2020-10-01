! This test checks lowering of OpenMP parallel Directive with arbitrary code
! inside it.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program parallel

        integer :: a,b,c
        integer :: num_threads

        a = 1
        b = 2
!FIRDialect:  %[[VAR_A:.*]] = fir.alloca i32 {name = "a"}
!FIRDialect:  %[[VAR_B:.*]] = fir.alloca i32 {name = "b"}
!FIRDialect:  %[[VAR_C:.*]] = fir.alloca i32 {name = "c"}
!FIRDialect:  %[[VAR_NUM_THREADS:.*]] = fir.alloca i32 {name = "num_threads"}

!LLVMIRDialect: %[[VAR_A:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "a"}
!LLVMIRDialect: %[[VAR_B:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "b"}
!LLVMIRDialect: %[[VAR_C:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "c"}
!LLVMIRDialect: %[[VAR_NUM_THREADS:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "num_threads"}

!LLVMIR: %[[OMP_GLOBAL_THREAD_NUM:.*]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.*}})
!LLVMIR: call void @__kmpc_push_num_threads(%struct.ident_t* @{{.*}}, i32 %[[OMP_GLOBAL_THREAD_NUM]], i32 %{{.*}})

!$OMP PARALLEL NUM_THREADS(num_threads)
!FIRDialect: omp.parallel num_threads(%{{.*}} : i32) {
!FIRDialect-DAG: %[[OMP_VAR_A:.*]] = fir.load %[[VAR_A]]
!FIRDialect-DAG: %[[OMP_VAR_B:.*]] = fir.load %[[VAR_B]]
!FIRDialect:     %[[OMP_VAR_C:.*]] = addi %[[OMP_VAR_A]], %[[OMP_VAR_B]]
!FIRDialect:     fir.store %[[OMP_VAR_C]] to %[[VAR_C]]
!FIRDialect-DAG:     %[[CONSTANT:.*]] = constant 4 : i32
!FIRDialect-DAG:     %[[COND_C:.*]] = fir.load %[[VAR_C]] : !fir.ref<i32>
!FIRDialect:     %[[COND_RES:.*]] = cmpi "sgt", %[[COND_C]], %[[CONSTANT]] : i32
!FIRDialect: fir.if %[[COND_RES]] {
!FIRDialect:       fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:       fir.call @_FortranAioOutputAscii
!FIRDialect:       fir.call @_FortranAioEndIoStatement
!FIRDialect:     } else {
!FIRDialect-NEXT:     }
!FIRDialect:     fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:     fir.load %[[VAR_C]]
!FIRDialect:     fir.call @_FortranAioOutputInteger64
!FIRDialect:     fir.call @_FortranAioEndIoStatement
!FIRDialect:     omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect-LABEL:   omp.parallel num_threads(%{{.*}} : !llvm.i32) {
!LLVMIRDialect-DAG: %[[OMP_VAR_A:.*]] = llvm.load %[[VAR_A:.*]]
!LLVMIRDialect-DAG: %[[OMP_VAR_B:.*]] = llvm.load %[[VAR_B:.*]]
!LLVMIRDialect:      %[[OMP_VAR_C:.*]] = llvm.add {{.*}}%[[OMP_VAR_A]]
!LLVMIRDialect:     llvm.store %[[OMP_VAR_C]], %[[VAR_C]]
!LLVMIRDialect:     %[[COND_C:.*]] = llvm.load %[[VAR_C]] : !llvm.ptr<i32>
!LLVMIRDialect:     %[[COND_RES:.*]] = llvm.icmp "sgt" %[[COND_C]], %{{.*}} : !llvm.i32
!LLVMIRDialect:        llvm.cond_br %[[COND_RES]], ^bb1, ^bb2
!LLVMIRDialect: ^bb1:  // pred: ^bb0
!LLVMIRDialect:     llvm.call @_FortranAioBeginExternalListOutput
!LLVMIRDialect:     llvm.call @_FortranAioOutputAscii
!LLVMIRDialect:     llvm.call @_FortranAioEndIoStatement
!LLVMIRDialect:     llvm.br ^bb2
!LLVMIRDialect: ^bb2:  // 2 preds: ^bb0, ^bb1
!LLVMIRDialect:     llvm.call @_FortranAioBeginExternalListOutput
!LLVMIRDialect:     llvm.load %[[VAR_C]] : !llvm.ptr<i32>
!LLVMIRDialect:     llvm.call @_FortranAioOutputInteger64
!LLVMIRDialect:     llvm.call @_FortranAioEndIoStatement
!LLVMIRDialect:     omp.terminator
!LLVMIRDialect-NEXT:   }

!LLVMIR: call {{.*}} @__kmpc_fork_call(%struct.ident_t* @{{.*}} @_QQmain..omp_par

!LLVMIR-LABEL: define internal void @_QQmain..omp_par
!LLVMIR: br label %[[REGION_1:.*]]
!LLVMIR: [[REGION_1]]:
!LLVMIR:  br label %[[REGION_1_1:.*]]
!LLVMIR: [[REGION_1_1]]:
!LLVMIR: %[[COND_RES:.*]] = icmp sgt i32 %{{.*}}, 4
!LLVMIR: br i1 %[[COND_RES]], label %{{.*}}, label %{{.*}}
!LLVMIR:   call i8* @_FortranAioBeginExternalListOutput
!LLVMIR:   call i1 @_FortranAioOutputInteger64
!LLVMIR:   call i32 @_FortranAioEndIoStatement
        c = a + b

        if (c .gt. 4) then
        print*, "Inside If Statement"
        endif

        print*, c

!$OMP END PARALLEL

!$OMP PARALLEL
        print*, "Second Region"
!FIRDialect: omp.parallel {
!FIRDialect: fir.call @_FortranAioBeginExternalListOutput
!FIRDialect: fir.call @_FortranAioOutputAscii
!FIRDialect: fir.call @_FortranAioEndIoStatement
!FIRDialect: omp.terminator
!FIRDialect-NEXT: }

!LLVMIRDialect: omp.parallel {
!LLVMIRDialect: llvm.call @_FortranAioBeginExternalListOutput
!LLVMIRDialect: llvm.call @_FortranAioOutputAscii
!LLVMIRDialect: llvm.call @_FortranAioEndIoStatement
!LLVMIRDialect: omp.terminator
!LLVMIRDialect:   }

!LLVMIR-DAG-LABEL: call {{.*}} @__kmpc_fork_call(%struct.ident_t* @{{.*}} @_QQmain..omp_par.1
!LLVMIR-DAG-LABEL: define internal void @_QQmain..omp_par.1
!LLVMIR:   call i8* @_FortranAioBeginExternalListOutput
!LLVMIR:   call i32 @_FortranAioEndIoStatement
!$OMP END PARALLEL

end program
