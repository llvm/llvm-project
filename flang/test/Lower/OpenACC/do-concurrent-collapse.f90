! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! COLLAPSE on a DO CONCURRENT with collapse value == control count (N==C).
! NVHPC extension; lowers like the equivalent nested-DO collapse.

subroutine dc_collapse_eq(a, n)
  integer :: n
  integer :: a(n,n,n)
  integer :: i, j, k
  !$acc parallel loop collapse(3)
  do concurrent (i=1:10, j=1:100, k=1:200)
    a(i,j,k) = a(i,j,k) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdc_collapse_eq(
! CHECK: acc.parallel combined(loop)
! CHECK: acc.loop combined(parallel)
! CHECK-SAME: control(%{{[a-z_0-9]+}} : i32, %{{[a-z_0-9]+}} : i32, %{{[a-z_0-9]+}} : i32)
! CHECK-SAME: to (%{{[a-z_0-9]+}}, %{{[a-z_0-9]+}}, %{{[a-z_0-9]+}} : i32, i32, i32)
! Body must be lowered into the acc.loop region (regression: must not be dropped).
! CHECK: hlfir.designate
! CHECK: arith.addi %{{.*}}, %{{.*}} : i32
! CHECK: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK: collapse = [3]

! collapse(force:N) on a DO CONCURRENT must lower the body too (force takes a
! different path in Bridge.cpp).

subroutine dc_collapse_force_eq(a, n)
  integer :: n
  integer :: a(n,n,n)
  integer :: i, j, k
  !$acc parallel loop collapse(force:3)
  do concurrent (i=1:10, j=1:100, k=1:200)
    a(i,j,k) = a(i,j,k) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdc_collapse_force_eq(
! CHECK: acc.loop combined(parallel)
! CHECK-SAME: control(%{{[a-z_0-9]+}} : i32, %{{[a-z_0-9]+}} : i32, %{{[a-z_0-9]+}} : i32)
! CHECK: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK: collapse = [3]
