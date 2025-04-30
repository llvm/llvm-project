!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -O0 -fopenmp -Hy,69,0x1000 -S -emit-llvm %s -o - | FileCheck %s
! RUN: %flang -i8 -O0 -fopenmp -Hy,69,0x1000 -S -emit-llvm %s -o - | FileCheck %s

module omp_task_alloc
  use, intrinsic :: iso_fortran_env, &
                    only: logical32
  implicit none
  logical(logical32) :: l32 = .false.

contains

subroutine foo
  implicit none
  logical(kind(l32)) :: x
  logical(kind(l32)) :: y

!$omp task
! // CHECK: call ptr @__kmpc_omp_task_alloc(ptr {{.+}}, i32 {{.+}}, i32 {{.+}}, i64 {{.+}}, i64 {{.+}}, ptr {{.+}})
  x = l32
!$omp end task

end subroutine

end module
