!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK-LABEL: distinct !DIGlobalVariable(name: "alc_arr"
!CHECK-SAME: type: [[TYPE:![0-9]+]]
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type,
!CHECK-SAME: elements: [[ELEMENTS:![0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), allocated: !DIExpression(DW_OP_push_object_address, DW_OP_deref)
!CHECK: [[ELEMENTS]] = !{[[ELEMENT:![0-9]+]]}
!CHECK: [[ELEMENT]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 96, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 136, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref, DW_OP_mul))

module mod_allocatable_arr
  integer, allocatable :: alc_arr(:)
end module

program main
  use mod_allocatable_arr
  allocate (alc_arr(10))
  alc_arr = 99
  print *, alc_arr
end program
