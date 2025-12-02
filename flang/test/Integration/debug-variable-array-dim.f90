! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

subroutine foo(a, n, m, p)
  integer n, m, p
  integer :: a(n:m, p)
  a(1, 2) = 10
  print *, a
end subroutine foo


! CHECK-DAG: ![[VAR0:.*]] = !DILocalVariable(name: "._QFfooEa3"{{.*}}scope: ![[SCOPE:[0-9]+]]{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: ![[VAR1:.*]] = !DILocalVariable(name: "._QFfooEa1"{{.*}}scope: ![[SCOPE]]{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: ![[VAR2:.*]] = !DILocalVariable(name: "._QFfooEa2"{{.*}}scope: ![[SCOPE]]{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: ![[SR1:.*]] = !DISubrange(count: ![[VAR1]], lowerBound: ![[VAR0]])
! CHECK-DAG: ![[SR2:.*]] = !DISubrange(count: ![[VAR2]])
! CHECK-DAG: ![[SR:.*]] = !{![[SR1]], ![[SR2]]}
! CHECK-DAG: ![[TY:.*]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[SR]])
! CHECK-DAG: !DILocalVariable(name: "a"{{.*}}scope: ![[SCOPE]]{{.*}}type: ![[TY]])

