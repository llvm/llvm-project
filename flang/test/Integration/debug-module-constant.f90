! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck --check-prefix=LINEONLY %s

! A named constant declared in a module is described like a module variable: in
! the scope of the module, with a linkage name, and visible outside this compile
! unit.

! CHECK-DAG: ![[FILE:.*]] = !DIFile(filename: {{.*}}debug-module-constant.f90{{.*}})
! CHECK-DAG: ![[CU:.*]] = distinct !DICompileUnit({{.*}}file: ![[FILE]]{{.*}})
! CHECK-DAG: ![[MOD:.*]] = !DIModule(scope: ![[CU]], name: "helper"{{.*}})
! CHECK-DAG: ![[I4:.*]] = !DIBasicType(name: "integer(kind=4)", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[R4:.*]] = !DIBasicType(name: "real(kind=4)", size: 32, encoding: DW_ATE_float)

module helper
! CHECK-DAG: ![[MAX:.*]] = distinct !DIGlobalVariable(name: "max_size", linkageName: "_QMhelperECmax_size", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+2]], type: ![[I4]], isLocal: false, isDefinition: true)
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[MAX]], expr: !DIExpression())
  integer, parameter :: max_size = 100

! CHECK-DAG: ![[PI:.*]] = distinct !DIGlobalVariable(name: "pi", linkageName: "_QMhelperECpi", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+2]], type: ![[R4]], isLocal: false, isDefinition: true)
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[PI]], expr: !DIExpression())
  real, parameter :: pi = 3.14159274

! CHECK-DAG: ![[PRIMES:.*]] = distinct !DIGlobalVariable(name: "primes", linkageName: "_QMhelperECprimes", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+3]], type: ![[ARR:.*]], isLocal: false, isDefinition: true)
! CHECK-DAG: ![[ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[I4]]{{.*}})
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[PRIMES]], expr: !DIExpression())
  integer, parameter :: primes(3) = [2, 3, 5]

! CHECK-DAG: ![[TAG:.*]] = distinct !DIGlobalVariable(name: "tag", linkageName: "_QMhelperECtag", scope: ![[MOD]], file: ![[FILE]], line: [[@LINE+3]], type: ![[STR:.*]], isLocal: false, isDefinition: true)
! CHECK-DAG: ![[STR]] = !DIStringType(size: 40, encoding: DW_ATE_ASCII)
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[TAG]], expr: !DIExpression())
  character(len=5), parameter :: tag = "hello"
end module helper

program test
  use helper
  implicit none
  integer :: n
  n = max_size + primes(2)
  print *, pi, tag, n
end program test

! LINEONLY-NOT: DIGlobalVariable
