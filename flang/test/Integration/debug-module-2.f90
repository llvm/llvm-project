! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck --check-prefix=LINEONLY %s

! CHECK-DAG: ![[FILE:.*]] = !DIFile(filename: {{.*}}debug-module-2.f90{{.*}})
! CHECK-DAG: ![[FILE2:.*]] = !DIFile(filename: {{.*}}debug-module-2.f90{{.*}})
! CHECK-DAG: ![[CU:.*]] = distinct !DICompileUnit({{.*}}file: ![[FILE]]{{.*}} globals: ![[GLOBALS:.*]])
! CHECK-DAG: ![[MOD:.*]] = !DIModule(scope: ![[CU]], name: "helper", file: ![[FILE]]{{.*}})
! CHECK-DAG: ![[R4:.*]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[I4:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
module helper
! CHECK-DAG: ![[GLR:.*]] = distinct !DIGlobalVariable(name: "glr", linkageName: "_QMhelperEglr", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+2]], type: ![[R4]], isLocal: false, isDefinition: true)
! CHECK-DAG: ![[GLRX:.*]] = !DIGlobalVariableExpression(var: ![[GLR]], expr: !DIExpression())
  real glr

! CHECK-DAG: ![[GLI:.*]] = distinct !DIGlobalVariable(name: "gli", linkageName: "_QMhelperEgli", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+2]], type: ![[I4]], isLocal: false, isDefinition: true)
! CHECK-DAG: ![[GLIX:.*]] = !DIGlobalVariableExpression(var: ![[GLI]], expr: !DIExpression())
  integer gli

  contains
!CHECK-DAG: !DISubprogram(name: "test", linkageName: "_QMhelperPtest", scope: ![[MOD]], file: ![[FILE2]], line: [[@LINE+1]]{{.*}}unit: ![[CU]])
    subroutine test()
    glr = 12.34
    gli = 67

    end subroutine
end module helper

program test
use helper
implicit none

  glr = 3.14
  gli = 2
  call test()

end program test

! CHECK-DAG: ![[GLOBALS]] = !{![[GLIX]], ![[GLRX]]}
! LINEONLY-NOT: DIGlobalVariable
