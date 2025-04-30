!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.declare(metadata ptr %assume, metadata [[ASSUME:![0-9]+]], metadata !DIExpression())
!CHECK: [[TYPE:![0-9]+]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEMS:![0-9]+]])
!CHECK: [[ELEMS]] = !{[[ELEM1:![0-9]+]], [[ELEM2:![0-9]+]], [[ELEM3:![0-9]+]], [[ELEM4:![0-9]+]]
!CHECK: [[ELEM1]] = !DISubrange(lowerBound: 1, upperBound: 5)
!CHECK: [[ELEM2]] = !DISubrange(lowerBound: 1, upperBound: [[N1:![0-9]+]])
!CHECK: [[N1]] = distinct !DILocalVariable
!CHECK: [[ELEM3]] = !DISubrange(lowerBound: [[N2:![0-9]+]], upperBound: 9)
!CHECK: [[N2]] = distinct !DILocalVariable
!CHECK: [[ELEM4]] = !DISubrange(lowerBound: [[N3:![0-9]+]], upperBound: [[N4:![0-9]+]])
!CHECK: [[N3]] = distinct !DILocalVariable
!CHECK: [[N4]] = distinct !DILocalVariable
subroutine sub(assume,n1,n2,n3,n4)
  integer(kind=4) :: assume(5,n1,n2:9,n3:n4)
  assume(1,1,1,1) = 7
end subroutine sub
