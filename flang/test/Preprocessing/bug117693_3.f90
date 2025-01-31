! RUN: %flang -fopenmp -E %s 2>&1 | FileCheck %s
!CHECK-NOT: DO I=1,100 !$OMP
program main
INTEGER::n
DO I=1,100 !$OMP 
ENDDO
END PROGRAM
