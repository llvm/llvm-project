! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module helper
  character(len=40) :: str
end module helper

program test
  use helper
  character(kind=4, len=8) :: first
  character(len=10) :: second
  first = '3.14 = π'
  second = 'Fortran'
  str = 'Hello World!'
end program test

! CHECK-DAG: !DIGlobalVariable(name: "str"{{.*}}type: ![[TY40:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY40]] = !DIStringType(size: 320, encoding: DW_ATE_ASCII)
! CHECK-DAG: !DILocalVariable(name: "first"{{.*}}type: ![[TY8:[0-9]+]])
! CHECK-DAG: ![[TY8]] = !DIStringType(size: 256, encoding: DW_ATE_UCS)
! CHECK-DAG: !DILocalVariable(name: "second"{{.*}}type: ![[TY10:[0-9]+]])
! CHECK-DAG: ![[TY10]] = !DIStringType(size: 80, encoding: DW_ATE_ASCII)