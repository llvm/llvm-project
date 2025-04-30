!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Reproducer for a flang1 crash in getict.
! Ensure that flang1 generates a flag for empty derived type to mark null subc
! when exporting typedef initialization to MOD file.

! RUN: %flang1 %s
! RUN: FileCheck %s --input-file=./sconst_with_null_subc.mod
! RUN: rm -f ./sconst_with_null_subc.mod
module sconst_with_null_subc
  implicit none
  type empty
  end type
  type(empty), parameter :: x = empty()
end module sconst_with_null_subc

!CHECK: {{^}}S [[PARAM_ALIAS:[0-9]+]] {{.*}} x$ac{{$}}
!CHECK: {{^}}A [[AST:[0-9]+]] {{.*}} [[PARAM_ALIAS]]
!CHECK: {{^}}Z
!CHECK: {{^}}J [[# @LINE - 6]] 1 1
!CHECK: {{^}}V [[AST]]
!CHECK-NEXT: {{^}}S {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 1
!CHECK: {{^}}Z
