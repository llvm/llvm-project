! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.5.1 Parallel

program openacc_parallel_validity

  implicit none

  integer :: i, j, b, gang_size, vector_size, worker_size
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  integer :: async1
  integer :: wait1, wait2
  real :: reduction_r
  logical :: reduction_l
  real(8), dimension(N, N) :: aa, bb, cc
  real(8), dimension(:), allocatable :: dd
  real(8), pointer :: p
  logical :: ifCondition = .TRUE.
  real(8), dimension(N) :: a, f, g, h

  !$acc parallel device_type(*) num_gangs(2)
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
  end do
  !$acc end parallel

  !$acc parallel async
  !$acc end parallel

  !$acc parallel async(1)
  !$acc end parallel

  !ERROR: At most one ASYNC clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel async(1) async(2)
  !$acc end parallel

  !$acc parallel async(1) device_type(nvidia) async(3)
  !$acc end parallel

  !ERROR: At most one ASYNC clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel async(1) device_type(nvidia) async(2) async(3)
  !$acc end parallel

  !$acc parallel async(async1)
  !$acc end parallel

  !$acc parallel wait
  !$acc end parallel

  !$acc parallel wait(1)
  !$acc end parallel

  !$acc parallel wait(wait1)
  !$acc end parallel

  !$acc parallel wait(1,2)
  !$acc end parallel

  !$acc parallel wait(wait1, wait2)
  !$acc end parallel

  !$acc parallel num_gangs(8)
  !$acc end parallel

  !ERROR: NUM_GANGS clause accepts a maximum of 3 arguments
  !$acc parallel num_gangs(1, 1, 1, 1)
  !$acc end parallel

  !$acc parallel num_workers(8)
  !$acc end parallel

  !$acc parallel vector_length(128)
  !$acc end parallel

  !$acc parallel if(.true.)
  !$acc end parallel

  !$acc parallel if(ifCondition)
  !$acc end parallel

  !$acc parallel self
  !$acc end parallel

  !$acc parallel self(.true.)
  !$acc end parallel

  !$acc parallel self(ifCondition)
  !$acc end parallel

  !$acc parallel copy(aa) copyin(bb) copyout(cc)
  !$acc end parallel

  !$acc parallel copy(aa, bb) copyout(zero: cc)
  !$acc end parallel

  !$acc parallel present(aa, bb) create(cc)
  !$acc end parallel

  !$acc parallel copyin(readonly: aa, bb) create(zero: cc)
  !$acc end parallel

  !$acc parallel deviceptr(aa, bb) no_create(cc)
  !$acc end parallel

  !ERROR: Argument `cc` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc parallel attach(dd, p, cc)
  !$acc end parallel

  !$acc parallel private(aa) firstprivate(bb, cc)
  !$acc end parallel

  !$acc parallel default(none)
  !$acc end parallel

  !$acc parallel default(present)
  !$acc end parallel

  !$acc parallel device_type(*)
  !$acc end parallel

  !$acc parallel device_type(default)
  !$acc end parallel

  !$acc parallel device_type(default, host)
  !$acc end parallel

  !ERROR: Clause PRIVATE is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !ERROR: Clause FIRSTPRIVATE is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !$acc parallel device_type(*) private(aa) firstprivate(bb)
  !$acc end parallel

  !$acc parallel device_type(*) async
  !$acc end parallel

  !$acc parallel device_type(*) wait
  !$acc end parallel

  !$acc parallel device_type(*) num_gangs(8)
  !$acc end parallel

  !$acc parallel device_type(*) async device_type(host) wait
  !$acc end parallel

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !$acc parallel device_type(*) if(.TRUE.)
  !$acc loop
  do i = 1, N
    a(i) = 3.14d0
  end do
  !$acc end parallel

  do i = 1, 100
    !$acc parallel
    !ERROR: CYCLE to construct outside of PARALLEL construct is not allowed
    if (i == 10) cycle
    !$acc end parallel
  end do

  !$acc parallel
  do i = 1, 100
    if (i == 10) cycle
  end do
  !$acc end parallel

  !ERROR: At most one NUM_GANGS clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel num_gangs(400) num_gangs(400)
  !$acc end parallel

  !ERROR: At most one NUM_GANGS clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel device_type(nvidia) num_gangs(400) num_gangs(200)
  !$acc end parallel

  !$acc parallel device_type(nvidia) num_gangs(400) device_type(radeon) num_gangs(200)
  !$acc end parallel

  !ERROR: At most one NUM_WORKERS clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel num_workers(8) num_workers(4)
  !$acc end parallel

  !ERROR: At most one NUM_WORKERS clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel device_type(nvidia) num_workers(8) num_workers(4)
  !$acc end parallel

  !$acc parallel device_type(nvidia) num_workers(8) device_type(radeon) num_workers(4)
  !$acc end parallel

  !ERROR: At most one VECTOR_LENGTH clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel vector_length(128) vector_length(124)
  !$acc end parallel

  !ERROR: At most one VECTOR_LENGTH clause can appear on the PARALLEL directive or in group separated by the DEVICE_TYPE clause
  !$acc parallel device_type(nvidia) vector_length(256) vector_length(128)
  !$acc end parallel

  !$acc parallel device_type(nvidia) vector_length(256) device_type(radeon) vector_length(128)
  !$acc end parallel

end program openacc_parallel_validity

subroutine acc_parallel_default_none
  integer :: i, l
  real :: a(10,10)
  l = 10  
  !$acc parallel default(none)
  !$acc loop
  !ERROR: The DEFAULT(NONE) clause requires that 'l' must be listed in a data-mapping clause
  do i = 1, l
    !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
    a(1,i) = 1
  end do
  !$acc end parallel

  !$acc data copy(a)
  !$acc parallel loop firstprivate(l) default(none)
  do i = 1, l
    a(1,i) = 1
  end do
  !$acc end parallel
  !$acc end data
end subroutine acc_parallel_default_none
