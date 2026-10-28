!RUN: %python %S/../test_errors.py %s %flang -fopenmp

subroutine f00
!$omp parallel
!ERROR: cancel-directive-name clause is required on CANCEL directive
!$omp cancel
!$omp end parallel
end

subroutine f01
!$omp parallel
!ERROR: At most one cancel-directive-name clause can appear on CANCEL directive
!$omp cancel parallel parallel
!$omp end parallel
end

subroutine f02
!$omp parallel
!ERROR: cancel-directive-name clause is required on CANCELLATION POINT directive
!$omp cancellation point
!$omp end parallel
end

subroutine f03
!$omp parallel
!ERROR: At most one cancel-directive-name clause can appear on CANCELLATION POINT directive
!$omp cancellation point parallel parallel
!$omp end parallel
end

subroutine do_nowait1
!$omp parallel
!$omp do nowait
  do i=1,2
!ERROR: The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause
    !$omp cancel do
  enddo
!$omp end do
!$omp end parallel
end subroutine

subroutine do_nowait2
!$omp parallel
!$omp do
  do i=1,2
!ERROR: The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause
    !$omp cancel do
  enddo
!$omp end do nowait
!$omp end parallel
end subroutine

subroutine do_ordered
!$omp parallel do ordered
  do i=1,2
!ERROR: The CANCEL construct cannot be nested inside of a worksharing construct with the ORDERED clause
    !$omp cancel do
  enddo
!$omp end parallel do
end subroutine

subroutine sections_nowait1
!$omp parallel
!$omp sections nowait
  !$omp section
!ERROR: The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause
    !$omp cancel sections
!$omp end sections
!$omp end parallel
end subroutine

subroutine sections_nowait2
!$omp parallel
!$omp sections
  !$omp section
!ERROR: The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause
    !$omp cancel sections
!$omp end sections nowait
!$omp end parallel
end subroutine
