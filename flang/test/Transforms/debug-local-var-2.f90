! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck --check-prefix=LINEONLY %s

! This tests checks the debug information for local variables in llvm IR.

! CHECK-LABEL: define void @_QQmain
! CHECK-DAG: %[[AL11:.*]] = alloca i32
! CHECK-DAG: %[[AL12:.*]] = alloca i64
! CHECK-DAG: %[[AL13:.*]] = alloca i8
! CHECK-DAG: %[[AL14:.*]] = alloca i32
! CHECK-DAG: %[[AL15:.*]] = alloca float
! CHECK-DAG: %[[AL16:.*]] = alloca double
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL11]], metadata ![[I4:.*]], metadata !DIExpression())
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL12]], metadata ![[I8:.*]], metadata !DIExpression())
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL13]], metadata ![[L1:.*]], metadata !DIExpression())
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL14]], metadata ![[L4:.*]], metadata !DIExpression())
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL15]], metadata ![[R4:.*]], metadata !DIExpression())
! CHECK-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL16]], metadata ![[R8:.*]], metadata !DIExpression())
! CHECK-LABEL: }

! CHECK-LABEL: define {{.*}}i64 @_QFPfn1
! CHECK-SAME: (ptr %[[ARG1:.*]], ptr %[[ARG2:.*]], ptr %[[ARG3:.*]])
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[ARG1]], metadata ![[A1:.*]], metadata !DIExpression())
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[ARG2]], metadata ![[B1:.*]], metadata !DIExpression())
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[ARG3]], metadata ![[C1:.*]], metadata !DIExpression())
! CHECK-DAG: %[[AL2:.*]] = alloca i64
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[AL2]], metadata ![[RES1:.*]], metadata !DIExpression())
! CHECK-LABEL: }

! CHECK-LABEL: define {{.*}}i32 @_QFPfn2
! CHECK-SAME: (ptr %[[FN2ARG1:.*]], ptr %[[FN2ARG2:.*]], ptr %[[FN2ARG3:.*]])
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[FN2ARG1]], metadata ![[A2:.*]], metadata !DIExpression())
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[FN2ARG2]], metadata ![[B2:.*]], metadata !DIExpression())
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[FN2ARG3]], metadata ![[C2:.*]], metadata !DIExpression())
! CHECK-DAG: %[[AL3:.*]] = alloca i32
! CHECK-DAG: tail call void @llvm.dbg.declare(metadata ptr %[[AL3]], metadata ![[RES2:.*]], metadata !DIExpression())
! CHECK-LABEL: }

program mn
! CHECK-DAG: ![[MAIN:.*]] = distinct !DISubprogram(name: "_QQmain", {{.*}})

! CHECK-DAG: ![[TYI32:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[TYI64:.*]] = !DIBasicType(name: "integer", size: 64, encoding: DW_ATE_signed)
! CHECK-DAG: ![[TYL8:.*]]  = !DIBasicType(name: "logical", size: 8, encoding: DW_ATE_boolean)
! CHECK-DAG: ![[TYL32:.*]] = !DIBasicType(name: "logical", size: 32, encoding: DW_ATE_boolean)
! CHECK-DAG: ![[TYR32:.*]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[TYR64:.*]] = !DIBasicType(name: "real", size: 64, encoding: DW_ATE_float)

! CHECK-DAG: ![[I4]] = !DILocalVariable(name: "i4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYI32]])
! CHECK-DAG: ![[I8]] = !DILocalVariable(name: "i8", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYI64]])
! CHECK-DAG: ![[R4]] = !DILocalVariable(name: "r4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYR32]])
! CHECK-DAG: ![[R8]] = !DILocalVariable(name: "r8", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYR64]])
! CHECK-DAG: ![[L1]] = !DILocalVariable(name: "l1", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYL8]])
! CHECK-DAG: ![[L4]] = !DILocalVariable(name: "l4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYL32]])
  integer(kind=4) :: i4
  integer(kind=8) :: i8
  real(kind=4) :: r4
  real(kind=8) :: r8
  logical(kind=1) :: l1
  logical(kind=4) :: l4

  i8 = fn1(i4, r8, l1)
  i4 = fn2(i8, r4, l4)
contains
! CHECK-DAG: ![[FN1:.*]] = distinct !DISubprogram(name: "fn1", {{.*}})
! CHECK-DAG: ![[A1]] = !DILocalVariable(name: "a1", arg: 1, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI32]])
! CHECK-DAG: ![[B1]] = !DILocalVariable(name: "b1", arg: 2, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYR64]])
! CHECK-DAG: ![[C1]] = !DILocalVariable(name: "c1", arg: 3, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYL8]])
! CHECK-DAG: ![[RES1]] = !DILocalVariable(name: "res1", scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI64]])
  function fn1(a1, b1, c1) result (res1)
    integer(kind=4), intent(in) :: a1
    real(kind=8), intent(in) :: b1
    logical(kind=1), intent(in) :: c1
    integer(kind=8) :: res1

    res1 = a1 + b1
  end function

! CHECK-DAG: ![[FN2:.*]] = distinct !DISubprogram(name: "fn2", {{.*}})
! CHECK-DAG: ![[A2]] = !DILocalVariable(name: "a2", arg: 1, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI64]])
! CHECK-DAG: ![[B2]] = !DILocalVariable(name: "b2", arg: 2, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYR32]])
! CHECK-DAG: ![[C2]] = !DILocalVariable(name: "c2", arg: 3, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYL32]])
! CHECK-DAG: ![[RES2]] = !DILocalVariable(name: "res2", scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI32]])
  function fn2(a2, b2, c2) result (res2)
    integer(kind=8), intent(in) :: a2
    real(kind=4), intent(in) :: b2
    logical(kind=4), intent(in) :: c2
    integer(kind=4) :: res2

    res2 = a2 + b2
  end function
end program

LINEONLY-NOT: DILocalVariable
