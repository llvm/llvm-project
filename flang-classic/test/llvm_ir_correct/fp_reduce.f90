! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang                                 -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=CHECK
! RUN: %flang -Ofast                          -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=FAST
! RUN: %flang -Mx,216,0x8                     -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=NSZ
! RUN: %flang -Mx,216,0x10                    -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=REASSOC
! RUN: %flang -Mx,216,0x8 -Mx,216,0x10        -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=NSZ_REASSOC
! RUN: %flang -Ofast -Mx,216,0x8 -Mx,216,0x10 -S -emit-llvm %s -o - | FileCheck %s  --check-prefixes=FAST

real function acc(arr,N)
  real arr
  integer N
  do i=1, N
    acc = acc + arr(i)
! CHECK-NOT: fadd fast
! FAST: fadd fast
! FAST-NOT: fadd nsz
! FAST-NOT: fadd reassoc
! NSZ: fadd nsz
! REASSOC: fadd reassoc
! NSZ_REASSOC: fadd reassoc nsz
! NSZ_REASSOC-NOT: fadd fast
  end do
end function
