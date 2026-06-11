! This test checks lowering of OpenMP declare reduction with non-trivial types

! RUN: not %flang_fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

module mymod
  type advancedtype
     integer(4)::myarray(10)
     integer(4)::val
     integer(4)::otherval
  end type advancedtype
  !CHECK: not yet implemented: declare reduction currently only supports trivial types, fixed-length CHARACTER, or derived types containing them
  !$omp declare reduction(myreduction: advancedtype: omp_out = omp_in) initializer(omp_priv = omp_orig)
end module mymod

program mymaxtest
  use mymod

end program

