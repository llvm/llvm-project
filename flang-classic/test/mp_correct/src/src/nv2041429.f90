!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

subroutine slave_bar_checkin(pbar, nthreads)
    INTEGER, INTENT(INOUT) :: pbar
    INTEGER, INTENT(IN) :: nthreads
    INTEGER :: bar
    !$omp atomic
    pbar = pbar + 1
    !$omp end atomic
    DO
        !$omp atomic read
        bar = pbar
        !$omp end atomic
        IF (bar .ge. nthreads) then
          EXIT
        end if
    ENDDO
end subroutine

integer pbar
pbar=0

!$omp parallel
call slave_bar_checkin(pbar,4)
!$omp end parallel

print *, "PASS"
end

