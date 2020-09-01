! RUN: %S/test_errors.sh %s %t %f18 -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.6.5 Data
!   2.5.1 Parallel
!   2.5.2 Kernels
!   2.5.3 Serial
!   2.9 Loop
!   2.13 Declare
!   2.14.3 Set
!   2.14.4 Update
!   2.15.1 Routine
!   2.11 Parallel Loop
!   2.11 Kernels Loop
!   2.11 Serial Loop

program openacc_clause_validity

  implicit none

  integer :: i, j, b, gang_size, vector_size, worker_size
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  real :: reduction_r
  logical :: reduction_l
  real(8), dimension(N, N) :: aa

  !ERROR: At least one clause is required on the DECLARE directive
  !$acc declare
  real(8), dimension(N) :: a

  !ERROR: At least one of ATTACH, COPYIN, CREATE clause must appear on the ENTER DATA directive
  !$acc enter data

  !ERROR: Only the READONLY modifier is allowed for the COPYIN clause on the ENTER DATA directive
  !$acc enter data copyin(zero: i)

  !ERROR: Only the ZERO modifier is allowed for the CREATE clause on the ENTER DATA directive
  !$acc enter data create(readonly: i)

  !ERROR: Only the ZERO modifier is allowed for the COPYOUT clause on the DATA directive
  !$acc data copyout(readonly: i)
  !$acc end data

  !ERROR: COPYOUT clause is not allowed on the ENTER DATA directive
  !$acc enter data copyin(i) copyout(i)

  !ERROR: At most one IF clause can appear on the DATA directive
  !$acc data copy(i) if(.true.) if(.true.)
  !$acc end data

  !ERROR: At least one of COPYOUT, DELETE, DETACH clause must appear on the EXIT DATA directive
  !$acc exit data

  !ERROR: At least one of USE_DEVICE clause must appear on the HOST_DATA directive
  !$acc host_data
  !$acc end host_data

  !ERROR: At least one of DEFAULT_ASYNC, DEVICE_NUM, DEVICE_TYPE clause must appear on the SET directive
  !$acc set

  !ERROR: At least one of ATTACH, COPY, COPYIN, COPYOUT, CREATE, DEFAULT, DEVICEPTR, NO_CREATE, PRESENT clause must appear on the DATA directive
  !$acc data
  !$acc end data

  !$acc data copyin(i)
  !$acc end data

  !$acc data copyin(i)
  !ERROR: Unmatched PARALLEL directive
  !$acc end parallel

  !$acc update device(i) device_type(*) async

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the UPDATE directive
  !$acc update device(i) device_type(*) if(.TRUE.)

  !$acc parallel
  !ERROR: INDEPENDENT and SEQ clauses are mutually exclusive and may not appear on the same LOOP directive
  !$acc loop seq independent
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop tile(2)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel loop tile(2)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop tile(2, 2)
  do i = 1, N
    do j = 1, N
      aa(i, j) = 3.14
    end do
  end do

  !$acc parallel device_type(*) num_gangs(2)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop independent
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop auto
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(10)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(vector_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(length: vector_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(10)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(worker_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(num: worker_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(gang_size, static:*)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size, static:*)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size, static: gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop private(b, a(:))
  do i = 1, N
    a(i) = b
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop tile(*)
  do i = 1, N
    a(i) = b
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop tile(2, 2)
  do i = 1, N
    do j = 1, N
      a(i) = b
    end do
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: The parameter of the COLLAPSE clause must be a constant positive integer expression
  !$acc loop collapse(-1)
  do i = 1, N
    do j = 1, N
      a(i) = 3.14 + j
    end do
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause PRIVATE is not allowed after clause DEVICE_TYPE on the LOOP directive
  !$acc loop device_type(*) private(i)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause GANG is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop gang seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !$acc parallel device_type(*) if(.TRUE.)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL LOOP directive
  !$acc parallel loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel loop

  !$acc kernels device_type(*) async
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the KERNELS directive
  !$acc kernels device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the KERNELS LOOP directive
  !$acc kernels loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels loop

  !$acc serial device_type(*) async
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the SERIAL directive
  !$acc serial device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the SERIAL LOOP directive
  !$acc serial loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END KERNELS LOOP directive
  !$acc end kernels loop

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END SERIAL LOOP directive
  !$acc end serial loop

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END PARALLEL LOOP directive
  !$acc end parallel loop

  !$acc parallel loop reduction(+: reduction_r)
  do i = 1, N
    reduction_r = a(i) + i
  end do

  !$acc parallel loop reduction(*: reduction_r)
  do i = 1, N
    reduction_r = reduction_r * (a(i) + i)
  end do

  !$acc parallel loop reduction(min: reduction_r)
  do i = 1, N
    reduction_r = min(reduction_r, a(i) * i)
  end do

  !$acc parallel loop reduction(max: reduction_r)
  do i = 1, N
    reduction_r = max(reduction_r, a(i) * i)
  end do

  !$acc parallel loop reduction(iand: b)
  do i = 1, N
    b = iand(b, c(i))
  end do

  !$acc parallel loop reduction(ior: b)
  do i = 1, N
    b = ior(b, c(i))
  end do

  !$acc parallel loop reduction(ieor: b)
  do i = 1, N
    b = ieor(b, c(i))
  end do

  !$acc parallel loop reduction(.and.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .and. e(i)
  end do

  !$acc parallel loop reduction(.or.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .or. e(i)
  end do

  !$acc parallel loop reduction(.eqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .eqv. e(i)
  end do

  !$acc parallel loop reduction(.neqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .neqv. e(i)
  end do

  !$acc kernels wait(1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(queues: 1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(devnum: 1: 1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(devnum: 1: queues: 1, 2) async(3)
  !$acc end kernels

  !$acc wait(1)
  !$acc wait(1, 2)

  !$acc wait(queues: 1)
  !$acc wait(queues: 1, 2)

  !$acc wait(devnum: 1: 3)
  !$acc wait(devnum: 1: 3, 4)

  !$acc wait(devnum: 1: queues: 3)
  !$acc wait(devnum: 1: queues: 3, 4)

 contains

   subroutine sub1(a)
     real :: a(:)
     !ERROR: At least one of GANG, SEQ, VECTOR, WORKER clause must appear on the ROUTINE directive
     !$acc routine
   end subroutine sub1

   subroutine sub2(a)
     real :: a(:)
     !ERROR: Clause NOHOST is not allowed after clause DEVICE_TYPE on the ROUTINE directive
     !$acc routine seq device_type(*) nohost
   end subroutine sub2

end program openacc_clause_validity
