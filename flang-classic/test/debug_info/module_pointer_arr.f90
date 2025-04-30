!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK-LABEL: distinct !DIGlobalVariable(name: "ptr_arr"
!CHECK-SAME: type: [[TYPE:![0-9]+]]
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type,
!CHECK-SAME: elements: [[ELEMENTS:![0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref)
!CHECK: [[ELEMENTS]] = !{[[ELEMENT:![0-9]+]]}
!CHECK: [[ELEMENT]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 96, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 136, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref, DW_OP_mul))

module mod_pointer_arr
  integer, pointer :: ptr_arr(:)
end module

program main
use mod_pointer_arr
  integer, target :: tgtarr(20)
  tgtarr(1:20:2) = 22
  tgtarr(2:20:2) = 33
  ptr_arr => tgtarr(1:20:2)
  print *, ptr_arr
end program
