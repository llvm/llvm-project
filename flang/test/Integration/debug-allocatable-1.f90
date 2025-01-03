! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine ff(n, m)
  integer n, m, i, j
  integer, allocatable :: ar1(:, :)
  real, allocatable :: sc

  allocate(ar1(n, m))
  allocate(sc)
  sc = 3.14

  print *, sc
  print *, ar1
end subroutine ff


! CHECK-DAG: !DILocalVariable(name: "ar1"{{.*}}type: ![[TY1:[0-9]+]])
! CHECK-DAG: ![[TY1]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS2:[0-9]+]]{{.*}}dataLocation{{.*}}allocated: !DIExpression(DW_OP_push_object_address, DW_OP_deref, DW_OP_lit0, DW_OP_ne))
! CHECK-DAG: ![[ELEMS2]] = !{![[ELEM1:[0-9]+]], ![[ELEM2:[0-9]+]]}
! CHECK-DAG: ![[ELEM1]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref))
! CHECK-DAG: ![[ELEM2]] = !DISubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref))
! CHECK-DAG: !DILocalVariable(name: "sc"{{.*}}type: ![[TY2:[0-9]+]])
! CHECK-DAG: ![[TY2]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[TY3:[0-9]+]]{{.*}})
! CHECK-DAG: ![[TY3]] = !DIBasicType(name: "real"{{.*}})
