!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK-DAG: ![[B3i1:[0-9]+]] = distinct !DIGlobalVariable(name: "b3i", scope: ![[BLK:[0-9]+]]
!CHECK-DAG: ![[B3j1:[0-9]+]] = distinct !DIGlobalVariable(name: "b3j", scope: ![[BLK]]
!CHECK-DAG: ![[BLK]] = distinct !DICommonBlock(scope: ![[SCOPE1:[0-9]+]]
!CHECK-DAG: ![[SCOPE1]] = distinct !DISubprogram(name: "bdata"

!CHECK-DAG: ![[B3i2:[0-9]+]] = distinct !DIGlobalVariable(name: "b3i", scope: ![[MOD:[0-9]+]]
!CHECK-DAG: ![[B3j2:[0-9]+]] = distinct !DIGlobalVariable(name: "b3j", scope: ![[MOD]]
!CHECK-DAG: ![[MOD]] = !DIModule({{.*}}, name: ".blockdata."

!CHECK-DAG: ![[B3i3:[0-9]+]] = distinct !DIGlobalVariable(name: "b3i", scope: ![[SUB:[0-9]+]]
!CHECK-DAG: ![[B3j3:[0-9]+]] = distinct !DIGlobalVariable(name: "b3j", scope: ![[SUB]]
!CHECK-DAG: ![[SUB]] = distinct !DICommonBlock(scope: ![[SCOPE2:[0-9]+]]
!CHECK-DAG: ![[SCOPE2]] = distinct !DISubprogram(name: "sub_block_data"

!CHECK-DAG: ![[REF1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3i1]]
!CHECK-DAG: ![[REF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3j1]]
!CHECK-DAG: ![[REF3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3i2]]
!CHECK-DAG: ![[REF4:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3j2]]
!CHECK-DAG: ![[REF5:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3i3]]
!CHECK-DAG: ![[REF6:[0-9]+]] = !DIGlobalVariableExpression(var: ![[B3j3]]
!CHECK-DAG: @blk_ = global{{.*}}![[REF1]]{{.*}}![[REF2]]{{.*}}![[REF3]]{{.*}}![[REF4]]{{.*}}![[REF5]]{{.*}}![[REF6]]

PROGRAM bdata
   integer b3i, b3j, adr
   COMMON/BLK/b3i, b3j
   b3i = 111
   b3j = 222
   CALL sub_block_data      ! BP_BEFORE_SUB
   print *,"End of program"
END
! BLOCK DATA
BLOCK DATA
integer b3i, b3j
COMMON/BLK/b3i, b3j
DATA b3i, b3j/11, 22/
END
! SUBROUTINE
SUBROUTINE sub_block_data
   integer b3i, b3j
   COMMON/BLK/b3i, b3j
   b3i = 1111; ! BP_SUB
   b3j = 2222; ! BP_SUB
END
