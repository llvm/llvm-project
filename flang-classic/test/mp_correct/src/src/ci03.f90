!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test copyin common blocks
!

integer :: foo,bar
common /CMA/foo,bar
!$omp threadprivate(/CMA/)

!$omp master
    foo = 10
    bar = 20
!$omp end master

!$omp parallel copyin(/CMA/) num_threads(4)
    foo = foo + 1
    bar = bar + 1
    call check(foo, 11, 1)
    call check(bar, 21, 1)
!$omp end parallel
    end
