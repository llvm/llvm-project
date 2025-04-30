! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test that at -O2, flang1 emits zero initialization for array assignment of
! constant zeros, but only when the LHS has a rank of 2 or more, or if its type
! is complex. See collapse_assignment() in tools/flang1/flang1exe/transfrm.c.
!
! RUN: %flang -O2 -S -emit-flang-llvm -o %t.ll %s
! RUN: FileCheck %s < %t.ll

program collapse_zero_assignment
  implicit none
  integer(4), dimension(2) :: i1
  integer(4), dimension(3, 3) :: i2
  integer(4), dimension(5, 5, 5) :: i3
  real(4), dimension(4) :: f1
  real(4), dimension(6, 6) :: f2
  real(8), dimension(7) :: d1
  real(8), dimension(8, 8) :: d2
  complex(8), dimension(9) :: c1
  complex(8), dimension(10, 10) :: c2

! CHECK: call void{{.*}} @f90_mzero4 (ptr {{.*}}, i64 9)
! CHECK: call void{{.*}} @f90_mzero4 (ptr {{.*}}, i64 125)
! CHECK: call void{{.*}} @f90_mzero4 (ptr {{.*}}, i64 36)
! CHECK: call void{{.*}} @f90_mzero8 (ptr {{.*}}, i64 64)
! CHECK: call void{{.*}} @f90_mzeroz16 (ptr {{.*}}, i64 9)
! CHECK: call void{{.*}} @f90_mzeroz16 (ptr {{.*}}, i64 100)

  i1 = 0
  i2 = 0
  i3 = 0
  f1 = 0.0_4
  f2 = 0.0_4
  d1 = 0.0_8
  d2 = 0.0_8
  c1 = (0.0, 0.0)
  c2 = (0.0, 0.0)
end program
