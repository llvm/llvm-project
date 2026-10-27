! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc

! Recovery excludes AUTO and DEVICE_TYPE groups. Preserve their diagnostics.
subroutine unrecovered_modes(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !ERROR: AUTO and SEQ clauses are mutually exclusive and may not appear on the same LOOP directive
  !$acc loop seq auto
  do i = 1, 10
    a(i) = i
  end do

  !ERROR: AUTO and SEQ clauses are mutually exclusive and may not appear on the same LOOP directive
  !ERROR: INDEPENDENT and SEQ clauses are mutually exclusive and may not appear on the same LOOP directive
  !$acc loop seq auto independent
  do i = 1, 10
    a(i) = i
  end do

  !ERROR: At most one INDEPENDENT clause can appear on the LOOP directive
  !$acc loop device_type(nvidia) independent independent
  do i = 1, 10
    a(i) = i
  end do

  !ERROR: INDEPENDENT and SEQ clauses are mutually exclusive and may not appear on the same LOOP directive
  !$acc loop seq device_type(nvidia) independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

subroutine unrecovered_routine
  !ERROR: INDEPENDENT clause is not allowed on the ROUTINE SEQ INDEPENDENT directive
  !$acc routine seq independent
end subroutine
