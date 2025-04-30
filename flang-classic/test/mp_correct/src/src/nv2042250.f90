!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RECURSIVE INTEGER*8 FUNCTION parallel_sum(a)

    INTEGER, INTENT(IN) :: a(:)

    INTEGER*8 :: x

    x = 11

    !$omp task shared(x) firstprivate(a)
    x = 1
    !$omp end task

    !$omp taskwait

    parallel_sum = x
END FUNCTION

INTEGER*8 FUNCTION full_sum(a)

    interface
RECURSIVE INTEGER*8 FUNCTION parallel_sum(a)
    INTEGER, INTENT(IN) :: a(:)
END FUNCTION
    end interface

    INTEGER, INTENT(IN) :: a(:)

    INTEGER*8 :: s

    !$omp parallel
        s = parallel_sum(a)
    !$omp end parallel
    
    full_sum = s

    RETURN
END FUNCTION

PROGRAM fortran_omp_taskwait

    interface
INTEGER*8 FUNCTION full_sum(a)
    INTEGER, INTENT(IN) :: a(:)
END FUNCTION
    end interface

    INTEGER, ALLOCATABLE :: a(:)

    INTEGER*8 :: s, expected
    INTEGER :: i

    PRINT *, ''

    ALLOCATE(a(N))

    s = full_sum(a)

    PRINT *, "PASS"

END PROGRAM fortran_omp_taskwait
