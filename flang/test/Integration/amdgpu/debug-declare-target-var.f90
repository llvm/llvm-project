! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp  -fopenmp-is-target-device -debug-info-kind=standalone %s -o - | FileCheck  %s

module helper
  implicit none
  real var_x
  real var_y
  !$omp declare target(var_x)
  !$omp declare target(var_y)
end module helper

subroutine init()
  use helper
  !$omp declare target
  var_x = 3.14
  var_y = 0.25
end

! CHECK-DAG: @_QMhelperEvar_x = addrspace(1) {{.*}}!dbg ![[XE:[0-9]+]]
! CHECK-DAG: @_QMhelperEvar_y = addrspace(1) {{.*}}!dbg ![[YE:[0-9]+]]
! CHECK-DAG: ![[XE]] = !DIGlobalVariableExpression(var: ![[X:[0-9]+]], expr: !DIExpression(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(1))))
! CHECK-DAG: ![[YE]] = !DIGlobalVariableExpression(var: ![[Y:[0-9]+]], expr: !DIExpression(DIOpArg(0, ptr addrspace(1)), DIOpDeref(ptr addrspace(1))))
! CHECK-DAG: ![[X]] = {{.*}}!DIGlobalVariable(name: "var_x"{{.*}})
! CHECK-DAG: ![[Y]] = {{.*}}!DIGlobalVariable(name: "var_y"{{.*}})
