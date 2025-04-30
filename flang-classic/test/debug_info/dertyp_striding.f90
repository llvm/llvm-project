!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK !DILocalVariable(name: "pvar", scope: {{![0-9]+}}, file: {{![0-9]+}}, type: {{![0-9]+}})
!CHECK: !DICompositeType(tag: DW_TAG_array_type, baseType: {{![0-9]+}}, size: 64, align: 64, elements: [[ELEM:![0-9]+]], dataLocation: {{![0-9]+}}, associated: {{![0-9]+}})
!CHECK: [[ELEM]] = !{[[ELEM1:![0-9]+]]}
!CHECK: [[ELEM1]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 80, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 120, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 24, DW_OP_deref, DW_OP_mul))
program main
    type dtype
        integer(kind=8) :: x
        integer(kind=8) :: y
        integer(kind=8) :: z
    end type
    type(dtype), dimension(10), target :: tvar
    integer(kind=8), dimension(:), pointer :: pvar => null()
    tvar(:)%x = 1
    tvar(:)%y = 2
    tvar(:)%z = 3
    pvar => tvar(1:9)%y
    print *, pvar
end program
