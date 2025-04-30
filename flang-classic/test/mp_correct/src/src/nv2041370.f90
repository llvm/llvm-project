!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CHECK(assertion) IF ( .NOT. (assertion)) THEN; PRINT *, 'Check failed on line ', __LINE__; STOP 1; ENDIF
!#define CHECK(assertion) IF ( .NOT. (assertion)) THEN; PRINT *, 'Check failed on line ', __LINE__; ENDIF

subroutine testCase6_taskyield()
    use omp_lib

    INTEGER :: nthreads
    INTEGER :: bar, pbar
    INTEGER :: selected_tid, tid
    INTEGER :: task_tid, task_tid0
    INTEGER :: i

    bar = 0
    task_tid = -1
    task_tid0 = -1

    selected_tid = 1

    !$omp parallel
        !$omp single
            nthreads = omp_get_num_threads()
            
            DO i = 1, nthreads
                !$omp task firstprivate(i, nthreads, selected_tid) private(pbar, tid)
                    tid = omp_get_thread_num()

                    if (tid == selected_tid) THEN
                        DO
                            !$omp atomic read
                            pbar = bar
                            !$omp end atomic
                            IF (pbar == nthreads - 1) EXIT
                        END DO

                        ! Now we know that all other threads are spinning waiting for
                        ! the bar value to become equal to nthreads. No one else can 
                        ! run the following task but this thread.
                        !$omp task
                            task_tid = tid
                        !$omp end task
                        task_tid0 = task_tid

                        ! Ensure that the task was actually deferred.
                        CHECK(task_tid == -1)
                        !PRINT *, task_tid !!!!!! UNCOMMENT TO FIX THE UNINITIALIZED VARIABLE FAILURE

                        ! Give the new task an explicit "go".
                        !$omp taskyield

                        ! Since all other threads are busy, this thread itself should
                        ! have run the task.
                        CHECK(task_tid == tid)

                        ! Release everybody.
                        !$omp atomic
                        bar = bar + 1
                        !$omp end atomic
                    ELSE
                        !$omp atomic
                        bar = bar + 1
                        !$omp end atomic

                        DO
                            !$omp atomic read
                            pbar = bar
                            !$omp end atomic
                            IF (pbar == nthreads) EXIT
                        END DO
                    ENDIF
                    print *, "PASS"
                !$omp end task
            END DO
        !$omp end single
    !$omp end parallel

    print *, 'task_tid0: ', task_tid0
end subroutine

program fortran_omp_task

call testCase6_taskyield

end program fortran_omp_task


