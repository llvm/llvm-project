! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! This is the same as collapse_zero_assignment.f90, but specialized for
! QuadFP types.
!
! REQUIRES: quadfp
! RUN: %flang -O2 -S -emit-flang-llvm -o %t.ll %s
! RUN: FileCheck %s < %t.ll

program collapse_zero_assignment_quadfp
  implicit none
  real(16), dimension(2) :: d1
  real(16), dimension(3, 3) :: d2
  real(16), dimension(4, 4, 4) :: d3
  complex(16), dimension(5) :: c1
  complex(16), dimension(6, 6) :: c2
  complex(16), dimension(7, 7, 7) :: c3

! CHECK: call void{{.*}} @f90_mzero16 (ptr {{.*}}, i64 9)
! CHECK: call void{{.*}} @f90_mzero16 (ptr {{.*}}, i64 64)
! CHECK: call void{{.*}} @f90_mzeroz32 (ptr {{.*}}, i64 5)
! CHECK: call void{{.*}} @f90_mzeroz32 (ptr {{.*}}, i64 36)
! CHECK: call void{{.*}} @f90_mzeroz32 (ptr {{.*}}, i64 343)

  d1 = 0.0
  d2 = 0.0
  d3 = 0.0
  c1 = (0.0, 0.0)
  c2 = (0.0, 0.0)
  c3 = (0.0, 0.0)
end program
