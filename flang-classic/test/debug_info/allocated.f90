!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DILocalVariable(name: "arr", scope: {{![0-9]+}}, file: {{![0-9]+}}, type: [[TYPE:![0-9]+]])
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEM:![0-9]+]], dataLocation: {{![0-9]+}}, allocated: {{![0-9]+}})
!CHECK: [[ELEM]] = !{[[ELEM1:![0-9]+]], [[ELEM2:![0-9]+]]}
!CHECK: [[ELEM1]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 80, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 120, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
!CHECK: [[ELEM2]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 168, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
program main
  integer(kind=4), allocatable :: arr(:, :)

  allocate (arr(10,10))
  arr(1,1) = 99
  print *, arr(1,1)
end program main
