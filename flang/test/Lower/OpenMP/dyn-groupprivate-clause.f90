! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s

! Lowering of the OpenMP 6.1 `dyn_groupprivate` clause on the directives that
! currently accept it in flang.

! test 0: target with bare size, no modifiers.
! CHECK-LABEL: func.func @_QPf00
! CHECK: omp.target dyn_groupprivate({{.*}} : i32)
subroutine f00(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(n)
  !$omp end target
end subroutine

! test 1: target with cgroup-only modifier.
! CHECK-LABEL: func.func @_QPf01
! CHECK: omp.target dyn_groupprivate(cgroup, {{.*}} : i32)
subroutine f01(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(cgroup: n)
  !$omp end target
end subroutine

! test 2: target with fallback(abort), no access-group.
! CHECK-LABEL: func.func @_QPf02
! CHECK: omp.target dyn_groupprivate(fallback(abort), {{.*}} : i32)
subroutine f02(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(fallback(abort): n)
  !$omp end target
end subroutine

! test 3: target with fallback(default_mem), no access-group.
! CHECK-LABEL: func.func @_QPf03
! CHECK: omp.target dyn_groupprivate(fallback(default_mem), {{.*}} : i32)
subroutine f03(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(fallback(default_mem): n)
  !$omp end target
end subroutine

! test 4: target with fallback(null), no access-group.
! CHECK-LABEL: func.func @_QPf04
! CHECK: omp.target dyn_groupprivate(fallback(null), {{.*}} : i32)
subroutine f04(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(fallback(null): n)
  !$omp end target
end subroutine

! test 5: target with cgroup + fallback(abort).
! CHECK-LABEL: func.func @_QPf05
! CHECK: omp.target dyn_groupprivate(cgroup, fallback(abort), {{.*}} : i32)
subroutine f05(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(cgroup, fallback(abort): n)
  !$omp end target
end subroutine

! test 6: target with cgroup + fallback(default_mem).
! CHECK-LABEL: func.func @_QPf06
! CHECK: omp.target dyn_groupprivate(cgroup, fallback(default_mem), {{.*}} : i32)
subroutine f06(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(cgroup, fallback(default_mem): n)
  !$omp end target
end subroutine

! test 7: target with cgroup + fallback(null).
! CHECK-LABEL: func.func @_QPf07
! CHECK: omp.target dyn_groupprivate(cgroup, fallback(null), {{.*}} : i32)
subroutine f07(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(cgroup, fallback(null): n)
  !$omp end target
end subroutine

! test 8: Constant integer literal as the size operand.
! CHECK-LABEL: func.func @_QPf08
! CHECK: %[[CST:.*]] = arith.constant 1024 : i32
! CHECK: omp.target dyn_groupprivate(%[[CST]] : i32)
subroutine f08()
  !$omp target dyn_groupprivate(1024)
  !$omp end target
end subroutine

! test 9: Arithmetic expression as the size operand (n*1024).
! CHECK-LABEL: func.func @_QPf09
! CHECK: %{{.*}} = arith.muli {{.*}} : i32
! CHECK: omp.target dyn_groupprivate({{.*}} : i32)
subroutine f09(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(n*1024)
  !$omp end target
end subroutine

! test 10: 64-bit (kind=8) integer size operand: verify the size type is propagated.
! CHECK-LABEL: func.func @_QPf10
! CHECK: omp.target dyn_groupprivate({{.*}} : i64)
subroutine f10(n)
  implicit none
  integer(kind=8) :: n
  !$omp target dyn_groupprivate(n)
  !$omp end target
end subroutine

! test 11: teams with bare size, no modifiers.
! CHECK-LABEL: func.func @_QPf11
! CHECK: omp.teams dyn_groupprivate({{.*}} : i32)
subroutine f11(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(n)
  x = 1
  !$omp end teams
end subroutine

! test 12: teams with cgroup-only modifier.
! CHECK-LABEL: func.func @_QPf12
! CHECK: omp.teams dyn_groupprivate(cgroup, {{.*}} : i32)
subroutine f12(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(cgroup: n)
  x = 1
  !$omp end teams
end subroutine

! test 13: teams with fallback(abort), no access-group.
! CHECK-LABEL: func.func @_QPf13
! CHECK: omp.teams dyn_groupprivate(fallback(abort), {{.*}} : i32)
subroutine f13(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(fallback(abort): n)
  x = 1
  !$omp end teams
end subroutine

! test 14: teams with fallback(default_mem), no access-group.
! CHECK-LABEL: func.func @_QPf14
! CHECK: omp.teams dyn_groupprivate(fallback(default_mem), {{.*}} : i32)
subroutine f14(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(fallback(default_mem): n)
  x = 1
  !$omp end teams
end subroutine

! test 15: teams with cgroup + fallback(default_mem).
! CHECK-LABEL: func.func @_QPf15
! CHECK: omp.teams dyn_groupprivate(cgroup, fallback(default_mem), {{.*}} : i32)
subroutine f15(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(cgroup, fallback(default_mem): n)
  x = 1
  !$omp end teams
end subroutine

! test 16: teams with cgroup + fallback(null).
! CHECK-LABEL: func.func @_QPf16
! CHECK: omp.teams dyn_groupprivate(cgroup, fallback(null), {{.*}} : i32)
subroutine f16(n)
  implicit none
  integer :: n
  integer :: x
  !$omp teams dyn_groupprivate(cgroup, fallback(null): n)
  x = 1
  !$omp end teams
end subroutine
