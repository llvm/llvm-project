!RUN: %flang %s -g -S -emit-llvm -o - | FileCheck %s

!Ensure that for an allocatable variable, we're taking the type of
!allocatable variable as DW_TAG_pointer_type.
!CHECK: call void @llvm.dbg.declare(metadata ptr %{{.*}}, metadata ![[DILocalVariable:[0-9]+]], metadata !DIExpression())
!CHECK: ![[DILocalVariable]] = !DILocalVariable(name: "alcvar"
!CHECK-SAME: type: ![[PTRTYPE:[0-9]+]]
!CHECK: ![[PTRTYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[TYPE:[0-9]+]]
!CHECK: ![[TYPE]] = !DIBasicType(name: "double precision",{{.*}}

program main
  real(kind=8), allocatable :: alcvar
  allocate(alcvar)
  alcvar = 7.7
  print *, alcvar
end program main
