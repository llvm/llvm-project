! RUN: %flang_fc1 -O1 -mllvm --enable-affine-opt -emit-llvm -fopenmp -o - %s \
! RUN: | FileCheck %s

!CHECK-LABEL: define void @foo_(ptr captures(none) %0) {{.*}} {
!CHECK: call void{{.*}}@__kmpc_fork_call{{.*}}@[[OMP_OUTLINED_FN_1:.*]])

subroutine foo(a)
  integer, dimension(100, 100), intent(out) :: a
  a = 1
end subroutine foo
