!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK !DILocalVariable(name: "ptr", scope: {{![0-9]+}}, file: {{![0-9]+}}, type: {{![0-9]+}})
!CHECK: !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEM:![0-9]+]], dataLocation: {{![0-9]+}}, associated: {{![0-9]+}})
!CHECK: [[ELEM]] = !{[[ELEM1:![0-9]+]], [[ELEM2:![0-9]+]]}
!CHECK: [[ELEM1]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 80, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 120, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
!CHECK: [[ELEM2]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 168, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
program main
  integer, target :: arr(10, 10)
  integer, pointer :: ptr(:, :)

  arr(1,1) = 99
  ptr => arr
  print *, ptr(1,1)
end program main
