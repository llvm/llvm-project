!Windows uses 0d+0a as line-endings, and the check fails.
!UNSUPPORTED: system-windows

!----------
! RUN lines
!----------
! Embed something that can be easily checked
! RUN: %flang_fc1 -emit-llvm -o - -fembed-offload-object=%S/Inputs/hello.f90 %s 2>&1 | FileCheck %s

! RUN: %flang_fc1 -emit-llvm-bc -o %t.bc %s 2>&1
! RUN: %flang_fc1 -emit-llvm -o - -fembed-offload-object=%S/Inputs/hello.f90 %t.bc 2>&1 | FileCheck %s

! CHECK: @[[OBJECT_1:.+]] = private constant [61 x i8] c"program hello\0A  write(*,*), \22Hello world!\22\0Aend program hello\0A", section ".llvm.offloading", align 8, !exclude !0
! CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @[[OBJECT_1]]], section "llvm.metadata"


! CHECK: !llvm.embedded.objects = !{![[METADATA_1:[0-9]+]]}
! CHECK: ![[METADATA_1]] = !{ptr @[[OBJECT_1]], !".llvm.offloading"}

parameter(i=1)
integer :: j
end program
