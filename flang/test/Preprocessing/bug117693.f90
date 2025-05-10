! RUN: %flang -fopenmp -E %s 2>&1 | FileCheck %s
!CHECK: !$OMP DO SCHEDULE(STATIC)
program main
IMPLICIT NONE
INTEGER:: I
#define OMPSUPPORT
!$ INTEGER :: omp_id
!$OMP PARALLEL DO
OMPSUPPORT !$OMP DO SCHEDULE(STATIC)
DO I=1,100
print *, omp_id
ENDDO
!$OMP END PARALLEL DO
end program
