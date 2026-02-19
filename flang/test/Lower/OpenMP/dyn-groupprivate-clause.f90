! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s

! Test lowering of dyn_groupprivate clause for target directive

! CHECK-LABEL: func.func @_QPf00
! CHECK: omp.target dyn_groupprivate({{.*}})
subroutine f00(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(n)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPf01
! CHECK: omp.target dyn_groupprivate(fallback(abort), {{.*}})
subroutine f01(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(fallback(abort): n)
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPf02
! CHECK: omp.target dyn_groupprivate(cgroup, fallback(default_mem), {{.*}})
subroutine f02(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(cgroup, fallback(default_mem): n)
  !$omp end target
end subroutine

! Test lowering of dyn_groupprivate clause for teams directive

! CHECK-LABEL: func.func @_QPf03
! CHECK: omp.teams dyn_groupprivate({{.*}})
subroutine f03(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(n)
  x = 1
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPf04
! CHECK: omp.teams dyn_groupprivate(cgroup, fallback(null), {{.*}})
subroutine f04(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(cgroup, fallback(null): n)
  x = 1
  !$omp end teams
end subroutine
