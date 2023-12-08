! RUN: %flang -E -fopenmp -fopenacc %s 2>&1 | FileCheck %s
! CHECK:  !$OMP parallel default(shared) private(super_very_long_name_for_the_va&
! CHECK:  !$OMP&riable)
! CHECK:  !$acc data copyin(super_very_long_name_for_the_variable, another_super&
! CHECK:  !$acc&_wordy_variable_to_test)
! Test correct continuations in compiler directives
subroutine foo
  integer :: super_very_long_name_for_the_variable
  integer :: another_super_wordy_variable_to_test

  super_very_long_name_for_the_variable = 42
  another_super_wordy_variable_to_test = super_very_long_name_for_the_variable * 2
  !$OMP parallel default(shared) private(super_very_long_name_for_the_variable)
  !$omp end parallel

  !$acc data copyin(super_very_long_name_for_the_variable, another_super_wordy_variable_to_test)
  !$acc end data
end subroutine foo
