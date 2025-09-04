!RUN: %flang_fc1 -emit-llvm -fopenmp -mmlir --force-no-alias=false %s -o - | FileCheck %s --check-prefix=NORT
!RUN: %flang_fc1 -emit-llvm -fopenmp -mmlir --force-no-alias=false %s -o - | FileCheck %s --check-prefix=LLVM

!Make sure that there are no calls to the mapper.
!NORT-NOT: call{{.*}}__tgt_target_data_begin_mapper
!NORT-NOT: call{{.*}}__tgt_target_data_end_mapper

!Make sure we generate the body
!LLVM: define internal void @_QFPf(ptr %[[A0:[0-9]+]], ptr %[[A1:[0-9]+]])
!LLVM:   %[[V0:[0-9]+]] = load i32, ptr %[[A0]], align 4
!LLVM:   %[[V1:[0-9]+]] = load i32, ptr %[[A1]], align 4
!LLVM:   %[[V2:[0-9]+]] = add i32 %[[V0]], %[[V1]]
!LLVM:   store i32 %[[V2]], ptr %[[A0]], align 4
!LLVM:   ret void
!LLVM: }


program test

call f(1, 2)

contains

subroutine f(x, y)
  integer :: x, y
  !$omp target data map(tofrom: x, y)
  x = x + y
  !$omp end target data
end subroutine
end
