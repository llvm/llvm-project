! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp  -fopenmp-is-target-device -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine fff(x, y)
  implicit none
  integer :: y(:)
  integer :: x

!$omp target map(tofrom: x) map(tofrom: y)
    x = 5
    y = 10
!$omp end target

end subroutine fff

!CHECK: define{{.*}}amdgpu_kernel void @[[FN:[0-9a-zA_Z_]+]]({{.*}}){{.*}}!dbg ![[SP:[0-9]+]]
!CHECK: #dbg_declare({{.*}}, ![[X:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), {{.*}})
!CHECK: #dbg_declare({{.*}}, ![[Y:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), {{.*}})
!CHECK: }

! CHECK-DAG: ![[SP]] = {{.*}}!DISubprogram(name: "[[FN]]"{{.*}})
! CHECK-DAG: ![[X]] = !DILocalVariable(name: "x", arg: 2, scope: ![[SP]]{{.*}}type: ![[INT:[0-9]+]])
! CHECK-DAG: ![[INT]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[Y]] = !DILocalVariable(name: "y", arg: 3, scope: ![[SP]]{{.*}}type: ![[ARR:[0-9]+]])
! CHECK-DAG: ![[ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]]{{.*}})
