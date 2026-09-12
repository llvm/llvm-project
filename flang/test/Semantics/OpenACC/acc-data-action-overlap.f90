! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Data-action clauses may overlap data-sharing clauses. Only two overlapping
! data-sharing clauses conflict.

subroutine data_actions_then_private(deviceptr_var)
  integer :: deviceptr_var
  integer :: copy_var, copyin_var, copyout_var, create_var, no_create_var
  integer :: present_var
  integer, pointer :: attach_var
  !$acc parallel copy(copy_var) copyin(copyin_var) copyout(copyout_var) &
  !$acc& create(create_var) no_create(no_create_var) present(present_var) &
  !$acc& deviceptr(deviceptr_var) attach(attach_var) &
  !$acc& private(copy_var, copyin_var, copyout_var, create_var, &
  !$acc& no_create_var, present_var, deviceptr_var, attach_var)
  copy_var = 1
  !$acc end parallel
end subroutine

subroutine firstprivate_then_data_actions(deviceptr_var)
  integer :: deviceptr_var
  integer :: copy_var, copyin_var, copyout_var, create_var, no_create_var
  integer :: present_var
  integer, pointer :: attach_var
  !$acc serial firstprivate(copy_var, copyin_var, copyout_var, create_var, &
  !$acc& no_create_var, present_var, deviceptr_var, attach_var) &
  !$acc& copy(copy_var) copyin(copyin_var) copyout(copyout_var) &
  !$acc& create(create_var) no_create(no_create_var) present(present_var) &
  !$acc& deviceptr(deviceptr_var) attach(attach_var)
  copy_var = 1
  !$acc end serial
end subroutine

subroutine reduction_with_data_actions(deviceptr_var)
  integer :: deviceptr_var
  integer :: copy_var, copyin_var, copyout_var, create_var, no_create_var
  integer :: present_var
  integer, allocatable :: attach_var
  !$acc parallel reduction(+:copy_var, copyin_var, copyout_var, create_var, &
  !$acc& no_create_var, present_var, deviceptr_var, attach_var) &
  !$acc& copy(copy_var) copyin(copyin_var) copyout(copyout_var) &
  !$acc& create(create_var) no_create(no_create_var) present(present_var) &
  !$acc& deviceptr(deviceptr_var) attach(attach_var)
  copy_var = copy_var + 1
  !$acc end parallel
end subroutine

subroutine overlapping_data_actions(deviceptr_var)
  integer :: deviceptr_var
  integer :: x
  integer, pointer :: p
  !$acc kernels copy(x) copyin(x) copyout(x) create(x) no_create(x) present(x)
  x = 1
  !$acc end kernels

  !$acc kernels deviceptr(deviceptr_var) copy(deviceptr_var)
  deviceptr_var = 1
  !$acc end kernels

  !$acc kernels attach(p) copy(p) present(p)
  p = 1
  !$acc end kernels
end subroutine

subroutine combined_data_action_private()
  integer :: i, x
  !$acc parallel loop copy(x) private(x)
  do i = 1, 10
    x = i
  end do

  !$acc serial loop private(x) present(x)
  do i = 1, 10
    x = i
  end do

  !$acc kernels loop create(x) private(x)
  do i = 1, 10
    x = i
  end do
end subroutine

subroutine common_block_data_action_private()
  integer :: a, b
  common /overlap_common/ a, b
  !$acc declare link(/overlap_common/)
  !$acc parallel copy(/overlap_common/) private(/overlap_common/)
  a = 1
  !$acc end parallel

  !$acc serial firstprivate(/overlap_common/) copyin(/overlap_common/)
  b = 1
  !$acc end serial
end subroutine
