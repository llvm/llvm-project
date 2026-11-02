! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -O2 -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=OPT
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=NOSLOT
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NOSLOT

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
! CHECK:         store i64 %{{[0-9]+}}, ptr %[[LB_SLOT]]
! CHECK-NEXT:    #dbg_declare(ptr %[[LB_SLOT]], ![[LB:[0-9]+]], !DIExpression(),
! CHECK-NEXT:    store i64 %{{[0-9]+}}, ptr %[[COUNT_SLOT]]
! CHECK-NEXT:    #dbg_declare(ptr %[[COUNT_SLOT]], ![[COUNT:[0-9]+]], !DIExpression(),
! CHECK-NOT:     #dbg_value
! Test that no fake.use is generated with an integer type i{{[0-9]+}}.
! CHECK-NOT:     @llvm.fake.use(i{{[0-9]+}} %
! CHECK:         ret void

! CHECK-LABEL: define void @char_len_(
! CHECK:         %[[LEN_SLOT:[0-9]+]] = alloca
! CHECK:         store i64 %{{[0-9]+}}, ptr %[[LEN_SLOT]]
! CHECK-NEXT:    #dbg_declare(ptr %[[LEN_SLOT]], ![[LEN:[0-9]+]], !DIExpression(),
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
! CHECK-DAG: ![[CTY]] = !DIStringType(stringLength: ![[LEN]], encoding: DW_ATE_ASCII, charType: !{{.*}})

! The slots are only made at -O0. Above it a bound is described by its value,
! as it was before, so a debugger has to find that value in a register.
! OPT-LABEL: define void @array_(
! OPT:         #dbg_value({{.*}}![[OPT_LB:[0-9]+]], !DIExpression
! OPT:         #dbg_value({{.*}}![[OPT_COUNT:[0-9]+]], !DIExpression
! OPT-LABEL: define void @char_len_(
! OPT:         #dbg_value({{.*}}![[OPT_LEN:[0-9]+]], !DIExpression

! OPT-DAG: ![[OPT_LB]] = !DILocalVariable(name: "._QFarrayEa2"{{.*}}flags: DIFlagArtificial)
! OPT-DAG: ![[OPT_COUNT]] = !DILocalVariable(name: "._QFarrayEa1"{{.*}}flags: DIFlagArtificial)
! OPT-DAG: ![[OPT_LEN]] = !DILocalVariable(name: "._QFchar_lenEc1"{{.*}}flags: DIFlagArtificial)

! A slot exists only to describe a variable, so neither is made when no
! variables are asked for.
! NOSLOT-NOT: #dbg_declare
! NOSLOT-NOT: #dbg_value
