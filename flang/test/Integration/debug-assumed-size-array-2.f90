! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

XFAIL: *

! Test that debug info for arrays with non constant extent is different from
! assumed size arrays.

module helper
  implicit none
  contains
  subroutine fn (a1, n)
    integer n
    integer  a1(5, n)
    print *, a1(1,1)
  end subroutine fn
end module helper

! CHECK-DAG: ![[TY1:[0-9]+]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS1:[0-9]+]]{{.*}})
! CHECK-DAG: ![[ELEMS1]] = !{![[ELM1:[0-9]+]], ![[ELM2:[0-9]+]]}
! CHECK-DAG: ![[ELM1]] = !DISubrange(count: 5, lowerBound: 1)
! CHECK-DAG: ![[ELM2]] = !DISubrange(count: [[VAR:[0-9]+]], lowerBound: 1)
! CHECK-DAG: ![[VAR]] = !DILocalVariable
