!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Reproducer for a flang1 crash in dinit_acl_val2.
! Ensure that flang1 successfully produces correct symbol table entries for the
! empty derived type S, the array A, as well as the array constructor [S()].

! RUN: %flang1 %s -opt 0 -q 0 1 -q 47 1 | FileCheck %s
! CHECK: datatype:[[STYPE:[0-9]+]] Derived member:1 size:0
! CHECK: datatype:[[ATYPE:[0-9]+]] Array type:[[STYPE]] dims:1
! CHECK: datatype:[[CTYPE:[0-9]+]] Array type:[[STYPE]] dims:1
! CHECK: Array Local dtype:[[ATYPE]]
! CHECK-SAME: 1:a
! CHECK: Array Static dtype:[[CTYPE]]
! CHECK-SAME: 8:z_c_3$ac

program aconst_with_null_subict
  type S
  end type S
  type(S), dimension(1) :: A = reshape([S()], [1])
  print *, A
end program aconst_with_null_subict
