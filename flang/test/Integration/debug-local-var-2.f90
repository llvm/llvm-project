! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -mllvm --write-experimental-debuginfo=false -o - | FileCheck %s --check-prefixes=BOTH,INTRINSICS
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -mllvm --write-experimental-debuginfo=true -o - | FileCheck %s --check-prefixes=BOTH,RECORDS
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -mllvm --write-experimental-debuginfo=false -o - | FileCheck --check-prefix=LINEONLY %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -mllvm --write-experimental-debuginfo=true -o - | FileCheck --check-prefix=LINEONLY %s

! This tests checks the debug information for local variables in llvm IR.

! BOTH-LABEL: define void @_QQmain
! BOTH-DAG: %[[AL16:.*]] = alloca double
! BOTH-DAG: %[[AL15:.*]] = alloca float
! BOTH-DAG: %[[AL14:.*]] = alloca i32
! BOTH-DAG: %[[AL13:.*]] = alloca i8
! BOTH-DAG: %[[AL12:.*]] = alloca i64
! BOTH-DAG: %[[AL11:.*]] = alloca i32
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL11]], metadata ![[I4:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL12]], metadata ![[I8:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL13]], metadata ![[L1:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL14]], metadata ![[L4:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL15]], metadata ![[R4:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL16]], metadata ![[R8:.*]], metadata !DIExpression())
! RECORDS-DAG: #dbg_declare(ptr %[[AL11]], ![[I4:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[AL12]], ![[I8:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[AL13]], ![[L1:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[AL14]], ![[L4:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[AL15]], ![[R4:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[AL16]], ![[R8:.*]], !DIExpression(), !{{.*}})
! BOTH-LABEL: }

! BOTH-LABEL: define {{.*}}i64 @_QFPfn1
! BOTH-SAME: (ptr %[[ARG1:.*]], ptr %[[ARG2:.*]], ptr %[[ARG3:.*]])
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[ARG1]], metadata ![[A1:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[ARG2]], metadata ![[B1:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[ARG3]], metadata ![[C1:.*]], metadata !DIExpression())
! RECORDS-DAG: #dbg_declare(ptr %[[ARG1]], ![[A1:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[ARG2]], ![[B1:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[ARG3]], ![[C1:.*]], !DIExpression(), !{{.*}})
! BOTH-DAG: %[[AL2:.*]] = alloca i64
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL2]], metadata ![[RES1:.*]], metadata !DIExpression())
! RECORDS-DAG: #dbg_declare(ptr %[[AL2]], ![[RES1:.*]], !DIExpression(), !{{.*}})
! BOTH-LABEL: }

! BOTH-LABEL: define {{.*}}i32 @_QFPfn2
! BOTH-SAME: (ptr %[[FN2ARG1:.*]], ptr %[[FN2ARG2:.*]], ptr %[[FN2ARG3:.*]])
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[FN2ARG1]], metadata ![[A2:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[FN2ARG2]], metadata ![[B2:.*]], metadata !DIExpression())
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[FN2ARG3]], metadata ![[C2:.*]], metadata !DIExpression())
! RECORDS-DAG: #dbg_declare(ptr %[[FN2ARG1]], ![[A2:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[FN2ARG2]], ![[B2:.*]], !DIExpression(), !{{.*}})
! RECORDS-DAG: #dbg_declare(ptr %[[FN2ARG3]], ![[C2:.*]], !DIExpression(), !{{.*}})
! BOTH-DAG: %[[AL3:.*]] = alloca i32
! INTRINSICS-DAG: call void @llvm.dbg.declare(metadata ptr %[[AL3]], metadata ![[RES2:.*]], metadata !DIExpression())
! RECORDS-DAG: #dbg_declare(ptr %[[AL3]], ![[RES2:.*]], !DIExpression(), !{{.*}})
! BOTH-LABEL: }

program mn
! BOTH-DAG: ![[MAIN:.*]] = distinct !DISubprogram(name: "_QQmain", {{.*}})

! BOTH-DAG: ![[TYI32:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! BOTH-DAG: ![[TYI64:.*]] = !DIBasicType(name: "integer", size: 64, encoding: DW_ATE_signed)
! BOTH-DAG: ![[TYL8:.*]]  = !DIBasicType(name: "logical", size: 8, encoding: DW_ATE_boolean)
! BOTH-DAG: ![[TYL32:.*]] = !DIBasicType(name: "logical", size: 32, encoding: DW_ATE_boolean)
! BOTH-DAG: ![[TYR32:.*]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! BOTH-DAG: ![[TYR64:.*]] = !DIBasicType(name: "real", size: 64, encoding: DW_ATE_float)

! BOTH-DAG: ![[I4]] = !DILocalVariable(name: "i4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYI32]])
! BOTH-DAG: ![[I8]] = !DILocalVariable(name: "i8", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYI64]])
! BOTH-DAG: ![[R4]] = !DILocalVariable(name: "r4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYR32]])
! BOTH-DAG: ![[R8]] = !DILocalVariable(name: "r8", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYR64]])
! BOTH-DAG: ![[L1]] = !DILocalVariable(name: "l1", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYL8]])
! BOTH-DAG: ![[L4]] = !DILocalVariable(name: "l4", scope: ![[MAIN]], file: !{{.*}}, line: [[@LINE+6]], type: ![[TYL32]])
  integer(kind=4) :: i4
  integer(kind=8) :: i8
  real(kind=4) :: r4
  real(kind=8) :: r8
  logical(kind=1) :: l1
  logical(kind=4) :: l4

  i8 = fn1(i4, r8, l1)
  i4 = fn2(i8, r4, l4)
contains
! BOTH-DAG: ![[FN1:.*]] = distinct !DISubprogram(name: "fn1", {{.*}})
! BOTH-DAG: ![[A1]] = !DILocalVariable(name: "a1", arg: 1, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI32]])
! BOTH-DAG: ![[B1]] = !DILocalVariable(name: "b1", arg: 2, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYR64]])
! BOTH-DAG: ![[C1]] = !DILocalVariable(name: "c1", arg: 3, scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYL8]])
! BOTH-DAG: ![[RES1]] = !DILocalVariable(name: "res1", scope: ![[FN1]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI64]])
  function fn1(a1, b1, c1) result (res1)
    integer(kind=4), intent(in) :: a1
    real(kind=8), intent(in) :: b1
    logical(kind=1), intent(in) :: c1
    integer(kind=8) :: res1

    res1 = a1 + b1
  end function

! BOTH-DAG: ![[FN2:.*]] = distinct !DISubprogram(name: "fn2", {{.*}})
! BOTH-DAG: ![[A2]] = !DILocalVariable(name: "a2", arg: 1, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI64]])
! BOTH-DAG: ![[B2]] = !DILocalVariable(name: "b2", arg: 2, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYR32]])
! BOTH-DAG: ![[C2]] = !DILocalVariable(name: "c2", arg: 3, scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYL32]])
! BOTH-DAG: ![[RES2]] = !DILocalVariable(name: "res2", scope: ![[FN2]], file: !{{.*}}, line: [[@LINE+5]], type: ![[TYI32]])
  function fn2(a2, b2, c2) result (res2)
    integer(kind=8), intent(in) :: a2
    real(kind=4), intent(in) :: b2
    logical(kind=4), intent(in) :: c2
    integer(kind=4) :: res2

    res2 = a2 + b2
  end function
end program

LINEONLY-NOT: DILocalVariable
