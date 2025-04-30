!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

subroutine testCase8_icv
    use omp_lib
    !$omp parallel
        !$omp task
            !$omp parallel
                PRINT *, omp_get_max_threads()
            !$omp end parallel
        !$omp end task
    !$omp end parallel
end subroutine
program fortran_omp_task
call testcase8_icv
print *, "PASS"
end program fortran_omp_task
