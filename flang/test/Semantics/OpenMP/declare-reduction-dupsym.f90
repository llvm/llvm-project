! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s

!! Check for duplicate symbol use.
subroutine dup_symbol()
  type :: loc
     integer :: x
     integer :: y
  end type loc
 
  integer :: my_red

!CHECK: error: Duplicate definition of 'my_red' in DECLARE REDUCTION
  !$omp declare reduction(my_red : loc :  omp_out%x = omp_out%x + omp_in%x) initializer(omp_priv%x = 0)
  
end subroutine dup_symbol
