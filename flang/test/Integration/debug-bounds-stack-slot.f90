! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

subroutine array(a, lb, n)
  integer :: lb, n
  real :: a(lb:n)
  a(lb) = 0.0
end subroutine

subroutine char_len(c)
  character(*) :: c
  c(1:1) = 'x'
end subroutine

! CHECK-LABEL: define void @array_(
! CHECK:         %[[LB_SLOT:[0-9]+]] = alloca
! CHECK:         %[[COUNT_SLOT:[0-9]+]] = alloca
! CHECK:         #dbg_declare(ptr %[[LB_SLOT]], ![[LB:[0-9]+]], !DIExpression(),
! CHECK:         #dbg_declare(ptr %[[COUNT_SLOT]], ![[COUNT:[0-9]+]], !DIExpression(),
! CHECK-NOT:     #dbg_value
! Test that no fake.use is generated with an integer type i{{[0-9]+}}.
! CHECK-NOT:     @llvm.fake.use(i{{[0-9]+}} %
! CHECK:         ret void

! CHECK-LABEL: define void @char_len_(
! CHECK:         %[[LEN_SLOT:[0-9]+]] = alloca
! CHECK:         #dbg_declare(ptr %[[LEN_SLOT]], ![[LEN:[0-9]+]], !DIExpression(),
! CHECK-NOT:     #dbg_value
! CHECK-NOT:     @llvm.fake.use(i{{[0-9]+}} %
! CHECK:         ret void

! CHECK-DAG: ![[LB]] = !DILocalVariable(name: "._QFarrayEa2"{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: ![[COUNT]] = !DILocalVariable(name: "._QFarrayEa1"{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: !DILocalVariable(name: "a"{{.*}}type: ![[ATY:[0-9]+]])
! CHECK-DAG: ![[ATY]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}elements: ![[ELEMS:[0-9]+]])
! CHECK-DAG: ![[ELEMS]] = !{![[SUBRANGE:[0-9]+]]}
! CHECK-DAG: ![[SUBRANGE]] = !DISubrange(count: ![[COUNT]], lowerBound: ![[LB]])
! CHECK-DAG: ![[LEN]] = !DILocalVariable(name: "._QFchar_lenEc1"{{.*}}flags: DIFlagArtificial)
! CHECK-DAG: !DILocalVariable(name: "c"{{.*}}type: ![[CTY:[0-9]+]])
! CHECK-DAG: ![[CTY]] = !DIStringType(stringLength: ![[LEN]], encoding: DW_ATE_ASCII)
