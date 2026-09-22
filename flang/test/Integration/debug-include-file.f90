! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -I %t %t/main.f90 -o - | FileCheck %s

! Anything read through an INCLUDE gets a location that is fused with the
! inclusion information. Check that a module, a module variable, a procedure,
! its dummy argument and an internal procedure all still get debug
! information, and that it points at the file and line where each is written
! rather than at the INCLUDE statement.

!--- body.f90
! Nothing here starts on line 1 on purpose. 1 is also the line that is reported
! when a position cannot be read out of the location, so checking for it would
! pass whether or not the location was understood.
module included_mod
  integer :: modvar = 7
end module

subroutine included_sub(i)
  integer :: i
  i = 1
  call inner()
contains
subroutine inner()
  i = i + 1
end subroutine
end subroutine

!--- main.f90
include 'body.f90'
program p
  use included_mod
  integer :: i
  call included_sub(i)
  print *, i, modvar
end program

! CHECK-DAG: ![[BODY:[0-9]+]] = !DIFile(filename: "body.f90"
! CHECK-DAG: ![[MOD:[0-9]+]] = !DIModule({{.*}}name: "included_mod", file: ![[BODY]], line: 4)
! CHECK-DAG: !DIGlobalVariable(name: "modvar", linkageName: "_QMincluded_modEmodvar", scope: ![[MOD]], file: ![[BODY]], line: 5
! CHECK-DAG: ![[SUB:[0-9]+]] = distinct !DISubprogram(name: "included_sub", linkageName: "included_sub_", {{.*}}file: ![[BODY]], line: 8, {{.*}}scopeLine: 8
! CHECK-DAG: !DILocalVariable(name: "i", arg: 1, scope: ![[SUB]], file: ![[BODY]], line: 9
! CHECK-DAG: !DISubprogram(name: "inner", linkageName: "_QFincluded_subPinner", scope: ![[SUB]], file: ![[BODY]], line: 13, {{.*}}scopeLine: 13
! CHECK-DAG: !DISubprogram(name: "p", linkageName: "_QQmain", {{.*}}file: ![[MAIN:[0-9]+]], line: 2
! CHECK-DAG: ![[MAIN]] = !DIFile(filename: "main.f90"
