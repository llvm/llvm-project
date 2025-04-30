!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -S -emit-llvm %s -o - 2>&1 | FileCheck %s

! CHECK: define void @m_sub_
! CHECK-NOT: define internal void @m_sub_

module m
  private
  type my_type
    integer :: x
   contains
    final :: sub
  end type
contains
  subroutine sub(t)
    type(my_type) :: t
  end
end
