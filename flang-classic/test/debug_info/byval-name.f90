!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!Verify the IR contains _V_ (flang)convention naming and the
! operation using them are well formed.
!CHECK: sub_(i32 [[PREFIXED_ARG_NAME:%_V_arg_abc.arg]])
!CHECK: [[PREFIXED_LOCAL_NAME:%_V_arg_abc.addr]] = alloca i32, align 4
!CHECK: call void @llvm.dbg.declare(metadata ptr [[PREFIXED_LOCAL_NAME]]
!CHECK: store i32 [[PREFIXED_ARG_NAME]], ptr [[PREFIXED_LOCAL_NAME]], align 4

!Verify the DebugInfo metadata contains prefix _V_ truncated names.
!CHECK: DILocalVariable(name: "arg_abc"
!CHECK-NOT: DILocalVariable(name: "_V_arg_abc"

subroutine sub(arg_abc)
      integer,value :: arg_abc
      integer :: abc_local
      abc_local = arg_abc
      print*, arg_abc
end subroutine
