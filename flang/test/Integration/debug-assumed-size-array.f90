! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module helper
  implicit none
  contains
  subroutine fn (a1, a2)
	  integer  a1(5, *), a2(*)
    print *, a1(1,1)
    print *, a2(2)
  end subroutine fn
end module helper

! CHECK-DAG: ![[TY1:[0-9]+]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS1:[0-9]+]]{{.*}})
! CHECK-DAG: ![[ELEMS1]] = !{![[ELM1:[0-9]+]], ![[EMPTY:[0-9]+]]}
! CHECK-DAG: ![[ELM1]] = !DISubrange(count: 5, lowerBound: 1)
! CHECK-DAG: ![[EMPTY]] = !DISubrange()
! CHECK-DAG: ![[TY2:[0-9]+]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS2:[0-9]+]]{{.*}})
! CHECK-DAG: ![[ELEMS2]] = !{![[EMPTY:[0-9]+]]}
! CHECK-DAG: !DILocalVariable(name: "a1"{{.*}}type: ![[TY1:[0-9]+]])
! CHECK-DAG: !DILocalVariable(name: "a2"{{.*}}type: ![[TY2:[0-9]+]])
