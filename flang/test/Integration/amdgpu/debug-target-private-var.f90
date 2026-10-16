! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-is-target-device -debug-info-kind=standalone %s -o - | FileCheck %s

! A variable private to a nested parallel or worksharing region is described by
! a record pointing to the cast of its alloca. That cast is the value captured
! when those regions are outlined, so the record travels into every outlined
! function and the variable stays visible in the innermost one.

subroutine fff(x)
  implicit none
  integer :: x
  integer :: i

!$omp target teams distribute parallel do map(tofrom: x) private(i)
  do i = 1, 10
    x = x + i
  end do
!$omp end target teams distribute parallel do

end subroutine fff

! CHECK: define internal void @{{.*}}..omp_par(i32 {{.*}}!dbg ![[SP:[0-9]+]] {
! CHECK: #dbg_declare(ptr %{{.*}}, ![[I:[0-9]+]], !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), {{.*}})
! CHECK: ![[SP]] = {{.*}}!DISubprogram(name: "{{.*}}..omp_par"{{.*}})
! CHECK: ![[I]] = !DILocalVariable(name: "i", scope: ![[SP]]{{.*}})
