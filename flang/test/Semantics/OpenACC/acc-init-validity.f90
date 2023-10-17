! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.14.1 Init

program openacc_init_validity

  implicit none

  integer :: i, j
  integer, parameter :: N = 256
  logical :: ifCondition = .TRUE.
  integer :: ifInt
  real :: ifReal
  real(8), dimension(N) :: a

  !$acc init
  !$acc init if(.TRUE.)
  !$acc init if(ifCondition)
  !$acc init if(ifInt)
  !$acc init device_num(1)
  !$acc init device_num(i)
  !$acc init device_type(i)
  !$acc init device_type(2, i, j)
  !$acc init device_num(i) device_type(i, j) if(ifCondition)

  !$acc parallel
  !ERROR: Directive INIT may not be called within a compute region
  !$acc init
  !$acc end parallel

  !$acc serial
  !ERROR: Directive INIT may not be called within a compute region
  !$acc init
  !$acc end serial

  !$acc kernels
  !ERROR: Directive INIT may not be called within a compute region
  !$acc init
  !$acc end kernels

  !$acc parallel
  !$acc loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc serial
  !$acc loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do
  !$acc end serial

  !$acc kernels
  !$acc loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do
  !$acc end kernels

  !$acc parallel loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do

  !$acc serial loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do

  !$acc kernels loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14
  end do

  !ERROR: At most one IF clause can appear on the INIT directive
  !$acc init if(.TRUE.) if(ifCondition)

  !ERROR: At most one DEVICE_NUM clause can appear on the INIT directive
  !$acc init device_num(1) device_num(i)

  !ERROR: At most one DEVICE_TYPE clause can appear on the INIT directive
  !$acc init device_type(2) device_type(i, j)

  !ERROR: Must have LOGICAL or INTEGER type
  !$acc init if(ifReal)

end program openacc_init_validity
