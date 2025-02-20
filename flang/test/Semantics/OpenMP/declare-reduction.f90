! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

program main
!CHECK-LABEL: MainProgram scope: main

  !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)

!CHECK: my_add_red: Misc ConstructName
  
end program main

