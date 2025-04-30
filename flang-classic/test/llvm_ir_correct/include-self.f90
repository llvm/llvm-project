!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -cpp %s -DSELF=\"`realpath "%s"`\" -S -emit-llvm -o - | FileCheck %s

#ifndef __SELF_INCLUDED
#define __SELF_INCLUDED

#include SELF

subroutine foo
! CHECK: foo
end subroutine

#endif
