!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "dtype", file: {{![0-9]+}}, size: {{[0-9]+}}, align: {{[0-9]+}}, elements: [[MEMS:![0-9]+]])
!CHECK: [[MEMS]] = !{[[MEM1:![0-9]+]], [[MEM2:![0-9]+]], [[MEM3:![0-9]+]]}
!CHECK: [[MEM1]] = !DIDerivedType(tag: DW_TAG_member, name: "arrptr1", scope: {{![0-9]+}}, file: {{![0-9]+}}, baseType: [[TYPE1:![0-9]+]]
!CHECK: [[TYPE1]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEM1:![0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
!CHECK: [[ELEM1]] = !{[[ELEM11:![0-9]+]], [[ELEM12:![0-9]+]]}
!CHECK: [[ELEM11]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 96, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 136, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref, DW_OP_mul))
!CHECK: [[ELEM12]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 144, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 184, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 176, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 40, DW_OP_deref, DW_OP_mul))
!CHECK: [[MEM2]] = !DIDerivedType(tag: DW_TAG_member, name: "arrptr2", scope: {{![0-9]+}}, file: {{![0-9]+}}, baseType: [[TYPE1:![0-9]+]]
!CHECK: [[MEM3]] = !DIDerivedType(tag: DW_TAG_member, name: "arralc", scope: {{![0-9]+}}, file: {{![0-9]+}}, baseType: [[TYPE2:![0-9]+]]
!CHECK: [[TYPE2]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEM1:![0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), allocated: !DIExpression(DW_OP_push_object_address, DW_OP_deref))

program main

  type dtype
     integer, pointer :: arrptr1(:,:)
     integer, pointer :: arrptr2(:,:)
     integer, allocatable :: arralc(:,:)
  end type dtype
  type(dtype) :: dvar1
  type(dtype), pointer :: dvar2

  allocate (dvar1%arrptr1 (5,5))

  dvar1%arrptr1 (1,1)= 9
  dvar1%arrptr1 (2,3)= 8
  print *, dvar1%arrptr1

  allocate (dvar1%arralc (3,2))
  dvar1%arralc (1,1)= 29
  dvar1%arralc (3,2)= 28
  print *, dvar1%arralc

  allocate (dvar2)
  allocate (dvar2%arrptr2 (3,4))
  dvar2%arrptr2 (1,1)= 19
  dvar2%arrptr2 (2,1)= 18
  dvar2%arrptr2 (2,3)= 17
  print *, dvar2%arrptr2

end program main
