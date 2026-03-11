! Test that argument addresses are placed on the stack to extend their lifetime
! when used for debug info.

! RUN: %flang_fc1 -emit-llvm -O0 -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine foo(x)
  real :: x
  call bar(x)
  call bazz()
end subroutine

! CHECK-LABEL:   void @foo_(
! CHECK-SAME:       %[[ARG:.*]])
! CHECK:         %[[VAL_0:.*]] = alloca ptr
! CHECK-NOT: !dbg
! CHECK:         store ptr %[[ARG]], ptr %[[VAL_0]]
! CHECK-NOT: !dbg
! CHECK:           #dbg_declare(ptr %[[VAL_0]], ![[X_DBG:.*]], !DIExpression(DW_OP_deref), !{{.*}})
! CHECK:         call void @bar_(ptr %[[ARG]]), !dbg !{{.*}}
! CHECK:         call void @bazz_(), !dbg !{{.*}}
! CHECK:         ret void, !dbg !{{.*}}

! CHECK: ![[X_DBG]] = !DILocalVariable(name: "x", arg: 1,
