! RUN: %flang -fopenmp -E %s 2>&1 | FileCheck %s
! CHECK: !$OMP PARALLEL DO
! CHECK: !$OMP END PARALLEL DO
program main
IMPLICIT NONE
INTEGER:: I
#define OMPSUPPORT
INTEGER :: omp_id
OMPSUPPORT !$OMP PARALLEL DO
DO I=1,100
print *, omp_id
ENDDO
OMPSUPPORT !$OMP END PARALLEL DO
end program
