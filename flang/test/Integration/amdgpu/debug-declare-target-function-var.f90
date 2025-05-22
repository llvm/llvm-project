! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp  -fopenmp-is-target-device -debug-info-kind=standalone %s -o - | FileCheck  %s

function add(a, b) result(ret)
  real ret
  real a
  real b
!$omp declare target
  if (a > b) then
    ret = a;
  else
    ret = b;
  end if
end

!CHECK: define float @add_({{.*}}){{.*}}!dbg ![[SP:[0-9]+]] {
!CHECK: #dbg_declare({{.*}}, ![[A:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), !{{.*}})
!CHECK: #dbg_declare({{.*}}, ![[B:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(ptr)), !{{.*}})
!CHECK: #dbg_declare({{.*}}, ![[RET:[0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(float)), !{{.*}})
!CHECK: }
!CHECK: ![[SP]] = {{.*}}!DISubprogram(name: "add"{{.*}})
!CHECK: ![[A]] = !DILocalVariable(name: "a", arg: 1, scope: ![[SP]]{{.*}})
!CHECK: ![[B]] = !DILocalVariable(name: "b", arg: 2, scope: ![[SP]]{{.*}})
!CHECK: ![[RET]] = !DILocalVariable(name: "ret", scope: ![[SP]]{{.*}})
