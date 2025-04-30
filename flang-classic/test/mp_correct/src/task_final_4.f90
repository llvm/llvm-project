!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!**    This test must be run with more than 1 thread.
       PROGRAM is_it_in_final
       
       use omp_lib

       integer n
       integer result(1)
       integer expect(1)
       parameter(n=1)
!$omp parallel
!$omp single
!$omp task
!$omp critical
       result(1) = omp_in_final()
!$omp end critical
!$omp end task
!$omp end single
!$omp end parallel

       call check(result,expect,n)

       END PROGRAM is_it_in_final
