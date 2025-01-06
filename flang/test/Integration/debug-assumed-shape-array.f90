! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine ff(arr, arr1)
  implicit none
    integer :: arr(:, :)
    integer :: arr1(3:, 4:)
    return arr(1,1) + arr1(3,4)
end subroutine ff

! CHECK-DAG: !DILocalVariable(name: "arr"{{.*}}type: ![[TY1:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY1]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS:[0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
! CHECK-DAG: ![[ELEMS]] = !{![[ELEM1:[0-9]+]], ![[ELEM2:[0-9]+]]}
! CHECK-DAG: ![[ELEM1]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 32, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref))
! CHECK-DAG: ![[ELEM2]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref))

! CHECK-DAG: !DILocalVariable(name: "arr1"{{.*}}type: ![[TY2:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY2]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS1:[0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
! CHECK-DAG: ![[ELEMS1]] = !{![[ELEM11:[0-9]+]], ![[ELEM12:[0-9]+]]}
! CHECK-DAG: ![[ELEM11]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 32, DW_OP_deref), lowerBound: 3, stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref))
! CHECK-DAG: ![[ELEM12]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref), lowerBound: 4, stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref))
