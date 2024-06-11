! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine ff(arr)
  implicit none
    integer :: arr(:, :)
    return arr(1,1)
end subroutine ff

! CHECK-DAG: !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS:[0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
! CHECK-DAG: ![[ELEMS]] = !{![[ELEM1:[0-9]+]], ![[ELEM2:[0-9]+]]}
! CHECK-DAG: ![[ELEM1]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 32, DW_OP_deref))
! CHECK-DAG: ![[ELEM2]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 48, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref))

