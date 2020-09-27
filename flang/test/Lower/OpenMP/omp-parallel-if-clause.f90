! This test checks lowering of OpenMP parallel Directive with
! `IF` clause present.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

!FIRDialect: %[[ALPHA:.*]] = fir.alloca i32 {name = "alpha"}
!FIRDialect:  %[[CONSTANT_4:.*]] = constant 4 : i32
!FIRDialect:  fir.store %[[CONSTANT_4]] to %[[ALPHA]] : !fir.ref<i32>
!FIRDialect:  %[[CONSTANT_0:.*]] = constant 0 : i32
!FIRDialect:  %[[LD_ALPHA:.*]] = fir.load %0 : !fir.ref<i32>
!FIRDialect:  %[[COND:.*]] = cmpi "sle", %[[LD_ALPHA]], %[[CONSTANT_0]] : i32
!FIRDialect:  omp.parallel if(%[[COND]] : i1) {
!FIRDialect:    omp.terminator
!FIRDialect:  }

!LLVMIRDialect:   %[[CONSTANT_4:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
!LLVMIRDialect:   %[[CONSTANT_0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
!LLVMIRDialect:   %[[ALPHA:.*]] = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "alpha"} : (!llvm.i64) -> !llvm.ptr<i32>
!LLVMIRDialect:   llvm.store %[[CONSTANT_4]], %[[ALPHA]] : !llvm.ptr<i32>
!LLVMIRDialect:   %[[LD_ALPHA:.*]] = llvm.load %[[ALPHA]] : !llvm.ptr<i32>
!LLVMIRDialect:   %[[COND:.*]] = llvm.icmp "sle" %[[LD_ALPHA]], %[[CONSTANT_0]] : !llvm.i32
!LLVMIRDialect:   omp.parallel if(%[[COND]] : !llvm.i1) {
!LLVMIRDialect:     omp.terminator
!LLVMIRDialect:   }

!LLVMIR:  %[[ALPHA:.*]] = alloca i32, i64 1
!LLVMIR:   store i32 4, i32* %[[ALPHA]], align 4
!LLVMIR:   %[[LD_ALPHA:.*]] = load i32, i32* %[[ALPHA]], align 4
!LLVMIR:   %[[COND:.*]] = icmp sle i32 %[[LD_ALPHA]], 0
!LLVMIR:   br i1 %[[COND]], label %[[PARALLEL:.*]], label %[[SERIAL:.*]]
!LLVMIR: [[PARALLEL]]:
!LLVMIR:   br label %omp_parallel
!LLVMIR: [[SERIAL]]:
!LLVMIR:   call void @__kmpc_serialized_parallel
!LLVMIR:   call void @_QQmain..omp_par
!LLVMIR:   call void @__kmpc_end_serialized_parallel

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
