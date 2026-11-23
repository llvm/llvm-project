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
  !$acc init device_type(default)
  !$acc init device_type(nvidia, radeon)
  !$acc init device_num(i) device_type(host, multicore) if(ifCondition)

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
    a(i) = 3.14d0
  end do
  !$acc end parallel

  !$acc serial
  !$acc loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14d0
  end do
  !$acc end serial

  !$acc kernels
  !$acc loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14d0
  end do
  !$acc end kernels

  !$acc parallel loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14d0
  end do

  !$acc serial loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14d0
  end do

  !$acc kernels loop
  do i = 1, N
    !ERROR: Directive INIT may not be called within a compute region
    !$acc init
    a(i) = 3.14d0
  end do

  !ERROR: At most one IF clause can appear on the INIT directive
  !$acc init if(.TRUE.) if(ifCondition)

  !ERROR: At most one DEVICE_NUM clause can appear on the INIT directive
  !$acc init device_num(1) device_num(i)

  ! OK
  !$acc init device_type(nvidia) device_type(default, *)

  !ERROR: Must have LOGICAL or INTEGER type
  !$acc init if(ifReal)

end program openacc_init_validity
