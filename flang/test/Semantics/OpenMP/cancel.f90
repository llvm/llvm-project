!RUN: %python %S/../test_errors.py %s %flang -fopenmp

subroutine f00
!$omp parallel
!ERROR: Missing cancel-directive-name clause on the CANCEL construct
!$omp cancel
!$omp end parallel
end

subroutine f01
!$omp parallel
!ERROR: Multiple cancel-directive-name clauses are not allowed on the CANCEL construct
!$omp cancel parallel parallel
!$omp end parallel
end

subroutine f02
!$omp parallel
!ERROR: Missing cancel-directive-name clause on the CANCELLATION POINT construct
!$omp cancellation point
!$omp end parallel
end

subroutine f03
!$omp parallel
!ERROR: Multiple cancel-directive-name clauses are not allowed on the CANCELLATION POINT construct
!$omp cancellation point parallel parallel
!$omp end parallel
end
