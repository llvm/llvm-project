! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang --target=aarch64-unknown-linux-gnu -Kieee -S -emit-llvm %s -o - | FileCheck %s
! RUN: %flang --target=aarch64-unknown-linux-gnu -Kieee -r8 -S -emit-llvm %s -o - | FileCheck %s
! REQUIRES: aarch64

subroutine testA(r1,r2,r3,r4,r5,r6)
! CHECK-LABEL: @testa
! CHECK: call float @__ps_sin_1(float
! CHECK: call double @__pd_sin_1(double
  real(kind=4) :: r1
  real(kind=4) :: r2
  real(kind=8) :: r3
  real(kind=8) :: r4

  r2 = sin(r1)*cos(r1)
  r4 = sin(r3)*cos(r3)

  r2 = sin(r1)
  r4 = sin(r3)
end subroutine

subroutine testB(r1,r2,r3,r4,r5,r6)
! CHECK-LABEL: @testb
! CHECK: call float @__ps_cos_1(float
! CHECK: call double @__pd_cos_1(double
  real(kind=4) :: r1
  real(kind=4) :: r2
  real(kind=8) :: r3
  real(kind=8) :: r4

  r2 = sin(r1)*cos(r1)
  r4 = sin(r3)*cos(r3)

  r2 = cos(r1)
  r4 = cos(r3)
end subroutine

! Ensure the math routines are declared like pure functions.

! CHECK: declare double @__pd_sin_1(double) #[[ATTR_ID:[0-9]+]]
! CHECK: declare double @__pd_cos_1(double) #[[ATTR_ID]]

! CHECK: declare float @__ps_sin_1(float) #[[ATTR_ID]]
! CHECK: declare float @__ps_cos_1(float) #[[ATTR_ID]]

! CHECK: attributes #[[ATTR_ID]] = { nounwind {{willreturn memory\(none\)|readnone willreturn}}
