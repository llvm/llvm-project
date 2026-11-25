! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Test that an associate-name that shadows a dummy argument or a SAVE variable
! is not visible after END ASSOCIATE.

subroutine shadow_dummy(r, n, c)
  integer, intent(in) :: r, n
  logical, intent(in) :: c
  if (c) then
    associate (r => n * 3)
      print *, r
    end associate
    print *, r
  end if
end subroutine shadow_dummy

subroutine shadow_save(n, c)
  integer, intent(in) :: n
  logical, intent(in) :: c
  integer, save :: r = 7
  if (c) then
    associate (r => n * 3)
      print *, r
    end associate
    print *, r
  end if
end subroutine shadow_save

! CHECK-DAG: ![[SPD:[0-9]+]] = distinct !DISubprogram(name: "shadow_dummy"
! CHECK-DAG: !DILocalVariable(name: "r", arg: 1, scope: ![[SPD]],
! CHECK-DAG: ![[BD:[0-9]+]] = distinct !DILexicalBlock(scope: ![[SPD]], {{.*}}line: 10,
! CHECK-DAG: !DILocalVariable(name: "r", scope: ![[BD]],
! CHECK-DAG: !DILocation(line: 11, column: 7, scope: ![[BD]])
! CHECK-DAG: !DILocation(line: 13, column: 5, scope: ![[SPD]])

! CHECK-DAG: ![[SPS:[0-9]+]] = distinct !DISubprogram(name: "shadow_save"
! CHECK-DAG: !DIGlobalVariable(name: "r", scope: ![[SPS]],
! CHECK-DAG: ![[BS:[0-9]+]] = distinct !DILexicalBlock(scope: ![[SPS]], {{.*}}line: 22,
! CHECK-DAG: !DILocalVariable(name: "r", scope: ![[BS]],
! CHECK-DAG: !DILocation(line: 23, column: 7, scope: ![[BS]])
! CHECK-DAG: !DILocation(line: 25, column: 5, scope: ![[SPS]])
