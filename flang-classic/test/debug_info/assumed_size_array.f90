!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.declare(metadata ptr %array1, metadata [[ARRAY1:![0-9]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.declare(metadata ptr %array2, metadata [[ARRAY2:![0-9]+]], metadata !DIExpression())
!CHECK: [[TYPE1:![0-9]+]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, align: 32, elements: [[ELEMS1:![0-9]+]])
!CHECK: [[ELEMS1]] = !{[[ELEM11:![0-9]+]]}
!CHECK: [[ELEM11]] = !DISubrange(lowerBound: 1)
!CHECK: [[TYPE2:![0-9]+]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, align: 32, elements: [[ELEMS2:![0-9]+]])
!CHECK: [[ELEMS2]] = !{[[ELEM21:![0-9]+]], [[ELEM22:![0-9]+]]}
!CHECK: [[ELEM21]] = !DISubrange(lowerBound: 4, upperBound: 9)
!CHECK: [[ELEM22]] = !DISubrange(lowerBound: 10)
!CHECK: [[ARRAY1]] = !DILocalVariable(name: "array1"
!CHECK-SAME: type: [[TYPE1]]
!CHECK: [[ARRAY2]] = !DILocalVariable(name: "array2"
!CHECK-SAME: type: [[TYPE2]]
subroutine sub (array1, array2)
  integer :: array1 (*)
  integer :: array2 (4:9, 10:*)

  array1(7:8) = 9
  array2(5, 10) = 10
end subroutine
