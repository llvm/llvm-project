! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Check that the simd simdlen(int) directive generates the correct metadata.
! RUN: %flang -fopenmp -O2 -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-00,CHECK-ALL
! RUN: %flang -fopenmp -O2 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=METADATA

subroutine func1(a, b, m)
! CHECK-ALL: define void @func1
  integer :: i, m, a(m), b(m)
  !$omp simd simdlen(8)
  do i = 1, m
    b(i) = a(i) + 1
  end do
! CHECK-00:      vector.ph:
! CHECK-00:      vector.body:
end subroutine func1

subroutine func2(a, b, m)
! CHECK-ALL: define void @func2
  integer :: i, m, a(m), b(m)
  !$omp simd simdlen(16)
  do i = 1, m
    b(i) = a(i) + 1
  end do
! CHECK-00:      vector.ph:
! CHECK-00:      vector.body:
end subroutine func2

! METADATA: !"llvm.loop.vectorize.enable", i1 true
! METADATA: !"llvm.loop.vectorize.width", i32 8
! METADATA: !"llvm.loop.vectorize.width", i32 16
! CHECK-00: load <[[VF:[0-9]+]] x i32>
! CHECK-00: store <[[VF]] x i32>
! CHECK-00: "llvm.loop.isvectorized", i32 1
! CHECK-00: "llvm.loop.unroll.runtime.disable"