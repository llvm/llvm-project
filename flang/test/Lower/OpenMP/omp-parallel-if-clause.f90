! This test checks lowering of OpenMP parallel Directive with
! `IF` clause present.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

!FIRDialect-LABEL: func @_QQmain() {
!FIRDialect-DAG: %[[ALPHA:.*]] = fir.alloca i32 {name = "{{.*}}Ealpha"}
!FIRDialect-DAG: %[[CONSTANT_4:.*]] = constant 4 : i32
!FIRDialect-DAG: fir.store %[[CONSTANT_4]] to %[[ALPHA]] : !fir.ref<i32>
!FIRDialect-DAG: %[[CONSTANT_0:.*]] = constant 0 : i32
!FIRDialect-DAG: %[[LD_ALPHA:.*]] = fir.load %[[ALPHA]] : !fir.ref<i32>
!FIRDialect:    %[[COND:.*]] = cmpi "sle", %[[LD_ALPHA]], %[[CONSTANT_0]] : i32
!FIRDialect:     omp.parallel if(%[[COND]] : i1) {
!FIRDialect:       omp.terminator
!FIRDialect:     }

!LLVMIRDialect-LABEL:   llvm.func @_QQmain() {
!LLVMIRDialect-DAG: %[[CONSTANT_4:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
!LLVMIRDialect-DAG: %[[CONSTANT_0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
!LLVMIRDialect-DAG: %[[ALPHA:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "{{.*}}Ealpha"} : (!llvm.i64) -> !llvm.ptr<i32>
!LLVMIRDialect-DAG: llvm.store %[[CONSTANT_4]], %[[ALPHA]] : !llvm.ptr<i32>
!LLVMIRDialect-DAG: %[[LD_ALPHA:.*]] = llvm.load %[[ALPHA]] : !llvm.ptr<i32>
!LLVMIRDialect:     %[[COND:.*]] = llvm.icmp "sle" %[[LD_ALPHA]], %[[CONSTANT_0]] : !llvm.i32
!LLVMIRDialect:     omp.parallel if(%[[COND]] : !llvm.i1) {
!LLVMIRDialect:       omp.terminator
!LLVMIRDialect:     }

!LLVMIR-LABEL: define void @_QQmain()
!LLVMIR-DAG: %[[ALPHA:.*]] = alloca i32, i64 1
!LLVMIR-DAG:  store i32 4, i32* %[[ALPHA]], align 4
!LLVMIR-DAG:  %[[LD_ALPHA:.*]] = load i32, i32* %[[ALPHA]], align 4
!LLVMIR-DAG:  %[[COND:.*]] = icmp sle i32 %[[LD_ALPHA]], 0
!LLVMIR:      br i1 %[[COND]], label %[[PARALLEL:.*]], label %[[SERIAL:.*]]
!LLVMIR:      [[PARALLEL]]:
!LLVMIR:      br label %omp_parallel
!LLVMIR:      [[SERIAL]]:
!LLVMIR:      call void @__kmpc_serialized_parallel
!LLVMIR:      call void @_QQmain..omp_par
!LLVMIR:      call void @__kmpc_end_serialized_parallel

program ifclause
        integer :: alpha
        alpha =  4

!$OMP PARALLEL IF(alpha .le. 0)
print*, "Equality statement: Execution: Serial"
!$OMP END PARALLEL

!$OMP PARALLEL IF(.false.)
print*, "False statement: Execution: Serial"
!$OMP END PARALLEL

!$OMP PARALLEL IF(alpha .ge. 0)
print*, "Equality statement: Execution: Parallel"
!$OMP END PARALLEL

!$OMP PARALLEL IF(.true.)
print*, "True statement: Execution: Parallel"
!$OMP END PARALLEL

end
