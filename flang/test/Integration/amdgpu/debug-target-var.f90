! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-is-target-device -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine fff(x, y)
  implicit none
  integer :: y(:)
  integer :: x

!$omp target map(tofrom: x) map(tofrom: y)
    x = 5
    y = 10
!$omp end target

end subroutine fff

! CHECK: define{{.*}}amdgpu_kernel void @[[FN:[0-9a-zA_Z_]+]](ptr %0, ptr %[[ARG1:[0-9]+]], ptr %[[ARG2:[0-9]+]]){{.*}}!dbg ![[SP:[0-9]+]]
! CHECK-DAG: store ptr %[[ARG1]], ptr %[[CAST1:[0-9]+]]{{.*}}
! CHECK-DAG: %[[CAST1]] = addrspacecast ptr addrspace(5) %[[AL1:[0-9]+]]
! CHECK-DAG: %[[AL1]] = alloca{{.*}}
! CHECK-DAG: store ptr %[[ARG2]], ptr %[[CAST2:[0-9]+]]{{.*}}
! CHECK-DAG: %[[CAST2]] = addrspacecast ptr addrspace(5) %[[AL2:[0-9]+]]
! CHECK-DAG: %[[AL2]] = alloca{{.*}}
! CHECK-DAG: #dbg_declare(ptr addrspace(5) %[[AL1]], ![[X:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), {{.*}})
! CHECK-DAG: #dbg_declare(ptr addrspace(5) %[[AL2]], ![[Y:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), {{.*}})
! CHECK: }

! CHECK-DAG: ![[SP]] = {{.*}}!DISubprogram(name: "[[FN]]"{{.*}})
! CHECK-DAG: ![[X]] = !DILocalVariable(name: "x", arg: 2, scope: ![[SP]]{{.*}}type: ![[INT:[0-9]+]])
! CHECK-DAG: ![[INT]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[Y]] = !DILocalVariable(name: "y", arg: 3, scope: ![[SP]]{{.*}}type: ![[ARR:[0-9]+]])
! CHECK-DAG: ![[ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]]{{.*}})
