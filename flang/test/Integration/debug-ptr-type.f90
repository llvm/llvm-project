! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine ff(n, m)
  implicit none
  integer i, j, m, n
  real(4), pointer :: par(:, :)
  integer, pointer :: psc
  integer, pointer :: par2(:)
  character(len=16), pointer :: pstr
  real(4), target :: ar(4, 5)
  integer, target :: sc
  integer, target, allocatable :: ar2(:)
  character(len=:), target, allocatable :: str

  str = 'Hello'
  pstr => str
  allocate(ar2(4))
  par2 => ar2
  do i=1,5
    do j=1,4
    ar(j,i) = 0.1
    par2(j) = j
    end do
  end do
  sc = 3
  psc => sc
  par => ar

  print *, sc
  print *, ar
  print *, ar2
  print *, str
  print *, psc
  print *, par
  print *, par2
  print *, pstr
end subroutine ff


! CHECK-DAG: ![[INT_TY:[0-9]+]] = !DIBasicType(name: "integer"{{.*}})
! CHECK-DAG: ![[ELEMS1:[0-9]+]] = !{!{{[0-9]+}}}
! CHECK-DAG: !DILocalVariable(name: "par"{{.*}}type: ![[ARR_TY1:[0-9]+]])
! CHECK-DAG: ![[ARR_TY1]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS2:[0-9]+]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne))
! CHECK-DAG: ![[ELEMS2]] = !{![[ELEM21:[0-9]+]], ![[ELEM22:[0-9]+]]}
! CHECK-DAG: ![[ELEM21]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref))
! CHECK-DAG: ![[ELEM22]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref))
! CHECK-DAG: !DILocalVariable(name: "par2"{{.*}}type: ![[ARR_TY2:[0-9]+]])
! CHECK-DAG: ![[ARR_TY2]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}, elements: ![[ELEMS1]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne))
! CHECK-DAG: !DILocalVariable(name: "psc"{{.*}}type: ![[PTR_TY:[0-9]+]])
! CHECK-DAG: ![[PTR_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[INT_TY]]{{.*}})
