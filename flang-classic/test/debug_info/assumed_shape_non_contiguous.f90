!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.value(metadata ptr %array, metadata [[ARRAYDL:![0-9]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.declare(metadata ptr %"array$sd", metadata [[ARRAY:![0-9]+]], metadata !DIExpression())
!CHECK-LABEL: distinct !DICompileUnit(language: DW_LANG_Fortran90,
!CHECK: [[ARRAY]] = !DILocalVariable(name: "array"
!CHECK-SAME: arg: 3
!CHECK-SAME: type: [[TYPE:![0-9]+]])
!CHECK: [[TYPE]] = !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 32, align: 32, elements: [[ELEM:![0-9]+]], dataLocation: [[ARRAYDL]])
!CHECK: [[ELEM]] = !{[[ELEM1:![0-9]+]], [[ELEM2:![0-9]+]]}
!CHECK: [[ELEM1]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 88, DW_OP_deref), lowerBound: 1, stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
!CHECK: [[ELEM2]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 136, DW_OP_deref), lowerBound: 1, stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))

subroutine show (message, array)
  character (len=*) :: message
  integer :: array(:,:)

  print *, message
  print *, array

end subroutine show

program test

  interface
     subroutine show (message, array)
       character (len=*) :: message
       integer :: array(:,:)
     end subroutine show
  end interface

  integer :: parray(4,4) = reshape((/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16/),(/4,4/))

  call show ("parray", parray(1:2,1:2))
end program test
