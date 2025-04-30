!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.value(metadata i64 99, metadata [[SPAR:![0-9]+]], metadata !DIExpression())
!CHECK: distinct !DIGlobalVariable(name: "apar"
!CHECK: [[SPAR]] = !DILocalVariable(name: "spar"

program main
  integer (kind=8) :: svar
  integer (kind=8) :: avar(5)
  integer (kind=8), parameter :: spar = 99
  integer (kind=8), parameter :: apar(5) = (/99, 98, 97, 96, 95/)
  svar = spar
  avar = apar

  print *, svar, avar, spar, apar

end program main
