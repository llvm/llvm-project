! RUN: %S/../test_errors.sh %s %t %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.14.2 Shutdown

program openacc_shutdown_validity

  implicit none

  integer :: i, j
  integer, parameter :: N = 256
  logical :: ifCondition = .TRUE.
  real(8), dimension(N) :: a

  !$acc parallel
  !ERROR: Directive SHUTDOWN may not be called within a compute region
  !$acc shutdown
  !$acc end parallel

  !$acc serial
  !ERROR: Directive SHUTDOWN may not be called within a compute region
  !$acc shutdown
  !$acc end serial

  !$acc kernels
  !ERROR: Directive SHUTDOWN may not be called within a compute region
  !$acc shutdown
  !$acc end kernels

  !$acc parallel
  !$acc loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc serial
  !$acc loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do
  !$acc end serial

  !$acc kernels
  !$acc loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do
  !$acc end kernels

  !$acc parallel loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do

  !$acc serial loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do

  !$acc kernels loop
  do i = 1, N
    !ERROR: Directive SHUTDOWN may not be called within a compute region
    !$acc shutdown
    a(i) = 3.14
  end do

  !$acc shutdown
  !$acc shutdown if(.TRUE.)
  !$acc shutdown if(ifCondition)
  !$acc shutdown device_num(1)
  !$acc shutdown device_num(i)
  !$acc shutdown device_type(i)
  !$acc shutdown device_type(2, i, j)
  !$acc shutdown device_num(i) device_type(i, j) if(ifCondition)

  !ERROR: At most one IF clause can appear on the SHUTDOWN directive
  !$acc shutdown if(.TRUE.) if(ifCondition)

  !ERROR: At most one DEVICE_NUM clause can appear on the SHUTDOWN directive
  !$acc shutdown device_num(1) device_num(i)

  !ERROR: At most one DEVICE_TYPE clause can appear on the SHUTDOWN directive
  !$acc shutdown device_type(2) device_type(i, j)

end program openacc_shutdown_validity
