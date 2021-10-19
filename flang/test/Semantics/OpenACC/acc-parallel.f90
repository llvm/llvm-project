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
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel async
  !$acc end parallel

  !$acc parallel async(1)
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

  !$acc parallel device_type(1)
  !$acc end parallel

  !$acc parallel device_type(1, 3)
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

  !$acc parallel device_type(1) async device_type(2) wait
  !$acc end parallel

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !$acc parallel device_type(*) if(.TRUE.)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

end program openacc_parallel_validity
