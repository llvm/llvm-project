! RUN: %python %S/../test_errors.py %s %flang -fopenacc

subroutine copy_then_reduction()
  integer :: x
  !$acc parallel copy(x) reduction(+:x)
  x = x + 1
  !$acc end parallel
end subroutine

subroutine reduction_then_copy()
  integer :: x
  !$acc parallel reduction(+:x) copy(x)
  x = x + 1
  !$acc end parallel
end subroutine

subroutine parallel_data_clauses_with_reduction()
  integer :: copy_var, copyin_var, copyout_var, create_var, no_create_var
  integer :: present_var
  !$acc parallel copy(copy_var) copyin(copyin_var) &
  !$acc& copyout(copyout_var) create(create_var) &
  !$acc& no_create(no_create_var) present(present_var) &
  !$acc& reduction(+:copy_var, copyin_var, copyout_var, create_var, &
  !$acc& no_create_var, present_var)
  copy_var = copy_var + 1
  copyin_var = copyin_var + 1
  copyout_var = copyout_var + 1
  create_var = create_var + 1
  no_create_var = no_create_var + 1
  present_var = present_var + 1
  !$acc end parallel
end subroutine

subroutine serial_reduction_then_data_clauses()
  integer :: copy_var, present_var
  !$acc serial reduction(+:copy_var, present_var) &
  !$acc& copy(copy_var) present(present_var)
  copy_var = copy_var + 1
  present_var = present_var + 1
  !$acc end serial
end subroutine

subroutine combined_copy_reduction()
  integer :: i, x
  !$acc parallel loop copy(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do

  !$acc serial loop reduction(+:x) copy(x)
  do i = 1, 10
    x = x + i
  end do

  !$acc kernels loop copy(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do
end subroutine

subroutine combined_present_reduction()
  integer :: i, x
  !$acc parallel loop present(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do

  !$acc serial loop reduction(+:x) present(x)
  do i = 1, 10
    x = x + i
  end do

  !$acc kernels loop present(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do
end subroutine
