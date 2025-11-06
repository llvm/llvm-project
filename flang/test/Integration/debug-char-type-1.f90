! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module helper
  character(len=40) :: str
  character(len=:), allocatable :: str2
end module helper

program test
  use helper
  character(kind=4, len=8) :: first
  character(len=10) :: second
  first = '3.14 = π'
  second = 'Fortran'
  str = 'Hello World!'
  str2 = 'A quick brown fox jumps over a lazy dog'
end program test

! CHECK-DAG: !DIGlobalVariable(name: "str"{{.*}}type: ![[TY40:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY40]] = !DIStringType(size: 320, encoding: DW_ATE_ASCII)
! CHECK-DAG: !DIGlobalVariable(name: "str2"{{.*}}type: ![[TY:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY]] = !DIStringType(stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8), stringLocationExpression: !DIExpression(DW_OP_push_object_address, DW_OP_deref), encoding: DW_ATE_ASCII)
! CHECK-DAG: !DILocalVariable(name: "first"{{.*}}type: ![[TY8:[0-9]+]])
! CHECK-DAG: ![[TY8]] = !DIStringType(size: 256, encoding: DW_ATE_UCS)
! CHECK-DAG: !DILocalVariable(name: "second"{{.*}}type: ![[TY10:[0-9]+]])
! CHECK-DAG: ![[TY10]] = !DIStringType(size: 80, encoding: DW_ATE_ASCII)
