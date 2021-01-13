! RUN: %S/test_errors.sh %s %t %f18 -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.6.5 Data
!   2.5.1 Parallel
!   2.5.2 Serial
!   2.5.3 Kernels
!   2.9 Loop
!   2.12 Atomic
!   2.14.3 Set
!   2.14.4 Update
!   2.15.1 Routine
!   2.10 Cache
!   2.11 Parallel Loop
!   2.11 Kernels Loop
!   2.11 Serial Loop
!   2.14.3 Set
!   2.14.1 Init
!   2.14.2 Shutdown
!   2.16.13 Wait

program openacc_clause_validity

  implicit none

  type atype
    real(8), dimension(10) :: arr
    real(8) :: s
  end type atype

  integer :: i, j, b, gang_size, vector_size, worker_size
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  integer :: async1
  integer :: wait1, wait2
  real :: reduction_r
  logical :: reduction_l
  real(8), dimension(N, N) :: aa, bb, cc
  logical :: ifCondition = .TRUE.
  type(atype) :: t
  type(atype), dimension(10) :: ta

  real(8), dimension(N) :: a, f, g, h

  !$acc init
  !$acc init if(.TRUE.)
  !$acc init if(ifCondition)
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

  !$acc parallel
  !ERROR: Directive SET may not be called within a compute region
  !$acc set default_async(i)
  !$acc end parallel

  !$acc serial
  !ERROR: Directive SET may not be called within a compute region
  !$acc set default_async(i)
  !$acc end serial

  !$acc kernels
  !ERROR: Directive SET may not be called within a compute region
  !$acc set default_async(i)
  !$acc end kernels

  !$acc parallel
  !$acc loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc serial
  !$acc loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do
  !$acc end serial

  !$acc kernels
  !$acc loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do
  !$acc end kernels

  !$acc parallel loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do

  !$acc serial loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do

  !$acc kernels loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do

  !ERROR: At least one of DEFAULT_ASYNC, DEVICE_NUM, DEVICE_TYPE clause must appear on the SET directive
  !$acc set

  !ERROR: At least one of DEFAULT_ASYNC, DEVICE_NUM, DEVICE_TYPE clause must appear on the SET directive
  !$acc set if(.TRUE.)

  !ERROR: At most one DEFAULT_ASYNC clause can appear on the SET directive
  !$acc set default_async(2) default_async(1)

  !ERROR: At most one DEFAULT_ASYNC clause can appear on the SET directive
  !$acc set default_async(2) default_async(1)

  !ERROR: At most one DEVICE_NUM clause can appear on the SET directive
  !$acc set device_num(1) device_num(i)

  !ERROR: At most one DEVICE_TYPE clause can appear on the SET directive
  !$acc set device_type(i) device_type(2, i, j)

  !$acc set default_async(2)
  !$acc set default_async(i)
  !$acc set device_num(1)
  !$acc set device_num(i)
  !$acc set device_type(i)
  !$acc set device_type(2, i, j)
  !$acc set device_num(1) default_async(2) device_type(2, i, j)
  !ERROR: At most one IF clause can appear on the INIT directive
  !$acc init if(.TRUE.) if(ifCondition)

  !ERROR: At most one DEVICE_NUM clause can appear on the INIT directive
  !$acc init device_num(1) device_num(i)

  !ERROR: At most one DEVICE_TYPE clause can appear on the INIT directive
  !$acc init device_type(2) device_type(i, j)

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

  !ERROR: At least one of ATTACH, COPYIN, CREATE clause must appear on the ENTER DATA directive
  !$acc enter data

  !ERROR: Modifier is not allowed for the COPYIN clause on the ENTER DATA directive
  !$acc enter data copyin(zero: i)

  !ERROR: Modifier is not allowed for the COPYOUT clause on the EXIT DATA directive
  !$acc exit data copyout(zero: i)

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

  !$acc host_data use_device(aa)
  !$acc end host_data

  !$acc host_data use_device(aa) if(.true.)
  !$acc end host_data

  !$acc host_data use_device(aa) if(ifCondition)
  !$acc end host_data

  !$acc host_data use_device(aa, bb) if_present
  !$acc end host_data

  !$acc host_data use_device(aa, bb) if(.true.) if_present
  !$acc end host_data

  !ERROR: At least one of DEFAULT_ASYNC, DEVICE_NUM, DEVICE_TYPE clause must appear on the SET directive
  !$acc set

  !ERROR: At least one of ATTACH, COPY, COPYIN, COPYOUT, CREATE, DEFAULT, DEVICEPTR, NO_CREATE, PRESENT clause must appear on the DATA directive
  !$acc data
  !$acc end data

  !$acc data copy(aa) if(.true.)
  !$acc end data

  !$acc data copy(aa) if(ifCondition)
  !$acc end data

  !$acc data copy(aa, bb, cc)
  !$acc end data

  !$acc data copyin(aa) copyin(readonly: bb) copyout(cc)
  !$acc end data

  !$acc data copyin(readonly: aa, bb) copyout(zero: cc)
  !$acc end data

  !$acc data create(aa, bb(:,:)) create(zero: cc(:,:))
  !$acc end data

  !$acc data no_create(aa) present(bb, cc)
  !$acc end data

  !$acc data deviceptr(aa) attach(bb, cc)
  !$acc end data

  !$acc data copy(aa, bb) default(none)
  !$acc end data

  !$acc data copy(aa, bb) default(present)
  !$acc end data

  !ERROR: At most one DEFAULT clause can appear on the DATA directive
  !$acc data copy(aa, bb) default(none) default(present)
  !$acc end data

  !ERROR: At most one IF clause can appear on the DATA directive
  !$acc data copy(aa) if(.true.) if(ifCondition)
  !$acc end data

  !$acc data copyin(i)
  !ERROR: Unmatched PARALLEL directive
  !$acc end parallel

  !ERROR: At least one of DEVICE, HOST, SELF clause must appear on the UPDATE directive
  !$acc update

  !$acc update self(a, f) host(g) device(h)

  !$acc update host(aa) async(1)

  !$acc update device(bb) async(async1)

  !ERROR: At most one ASYNC clause can appear on the UPDATE directive
  !$acc update host(aa, bb) async(1) async(2)

  !$acc update self(bb, cc(:)) wait(1)

  !ERROR: SELF clause on the UPDATE directive must have a var-list
  !$acc update self

  !$acc update device(aa, bb, cc) wait(wait1)

  !$acc update host(aa) host(bb) device(cc) wait(1,2)

  !$acc update device(aa, cc) wait(wait1, wait2)

  !$acc update device(aa) device_type(*) async

  !$acc update host(bb) device_type(*) wait

  !$acc update self(cc) device_type(1,2) async device_type(3) wait

  !ERROR: At most one IF clause can appear on the UPDATE directive
  !$acc update device(aa) if(.true.) if(ifCondition)

  !ERROR: At most one IF_PRESENT clause can appear on the UPDATE directive
  !$acc update device(bb) if_present if_present

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
  !ERROR: SEQ and AUTO clauses are mutually exclusive and may not appear on the same LOOP directive
  !$acc loop auto seq
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

  !$acc parallel loop self
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: SELF clause on the PARALLEL LOOP directive only accepts optional scalar logical expression
  !$acc parallel loop self(bb, cc(:))
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop self(.true.)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop self(ifCondition)
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
  !ERROR: At most one VECTOR clause can appear on the LOOP directive
  !$acc loop vector vector(128)
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
  !ERROR: At most one WORKER clause can appear on the LOOP directive
  !$acc loop worker worker(10)
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
  !ERROR: At most one GANG clause can appear on the LOOP directive
  !$acc loop gang gang(gang_size)
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

  !$acc parallel attach(aa, bb, cc)
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

  !$acc parallel
  !ERROR: Clause WORKER is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop worker seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause VECTOR is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop vector seq
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

  !$acc serial
  !$acc end serial

  !$acc serial async
  !$acc end serial

  !$acc serial async(1)
  !$acc end serial

  !ERROR: At most one ASYNC clause can appear on the SERIAL directive
  !$acc serial async(1) async(2)
  !$acc end serial

  !$acc serial async(async1)
  !$acc end serial

  !$acc serial wait
  !$acc end serial

  !$acc serial wait(1)
  !$acc end serial

  !$acc serial wait(wait1)
  !$acc end serial

  !$acc serial wait(1,2)
  !$acc end serial

  !$acc serial wait(wait1, wait2)
  !$acc end serial

  !$acc serial wait(wait1) wait(wait2)
  !$acc end serial

  !ERROR: NUM_GANGS clause is not allowed on the SERIAL directive
  !$acc serial num_gangs(8)
  !$acc end serial

  !ERROR: NUM_WORKERS clause is not allowed on the SERIAL directive
  !$acc serial num_workers(8)
  !$acc end serial

  !ERROR: VECTOR_LENGTH clause is not allowed on the SERIAL directive
  !$acc serial vector_length(128)
  !$acc end serial

  !$acc serial if(.true.)
  !$acc end serial

  !ERROR: At most one IF clause can appear on the SERIAL directive
  !$acc serial if(.true.) if(ifCondition)
  !$acc end serial

  !$acc serial if(ifCondition)
  !$acc end serial

  !$acc serial self
  !$acc end serial

  !$acc serial self(.true.)
  !$acc end serial

  !$acc serial self(ifCondition)
  !$acc end serial

  !$acc serial loop reduction(+: reduction_r)
  do i = 1, N
    reduction_r = a(i) + i
  end do

  !$acc serial loop reduction(*: reduction_r)
  do i = 1, N
    reduction_r = reduction_r * (a(i) + i)
  end do

  !$acc serial loop reduction(min: reduction_r)
  do i = 1, N
    reduction_r = min(reduction_r, a(i) * i)
  end do

  !$acc serial loop reduction(max: reduction_r)
  do i = 1, N
    reduction_r = max(reduction_r, a(i) * i)
  end do

  !$acc serial loop reduction(iand: b)
  do i = 1, N
    b = iand(b, c(i))
  end do

  !$acc serial loop reduction(ior: b)
  do i = 1, N
    b = ior(b, c(i))
  end do

  !$acc serial loop reduction(ieor: b)
  do i = 1, N
    b = ieor(b, c(i))
  end do

  !$acc serial loop reduction(.and.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .and. e(i)
  end do

  !$acc serial loop reduction(.or.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .or. e(i)
  end do

  !$acc serial loop reduction(.eqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .eqv. e(i)
  end do

  !$acc serial loop reduction(.neqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .neqv. e(i)
  end do

  !$acc serial reduction(.neqv.: reduction_l)
  !$acc loop reduction(.neqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .neqv. e(i)
  end do
  !$acc end serial

  !$acc serial copy(aa) copyin(bb) copyout(cc)
  !$acc end serial

  !$acc serial copy(aa, bb) copyout(zero: cc)
  !$acc end serial

  !$acc serial present(aa, bb) create(cc)
  !$acc end serial

  !$acc serial copyin(readonly: aa, bb) create(zero: cc)
  !$acc end serial

  !$acc serial deviceptr(aa, bb) no_create(cc)
  !$acc end serial

  !$acc serial attach(aa, bb, cc)
  !$acc end serial

  !$acc serial firstprivate(bb, cc)
  !$acc end serial

  !$acc serial private(aa)
  !$acc end serial

  !$acc serial default(none)
  !$acc end serial

  !$acc serial default(present)
  !$acc end serial

  !ERROR: At most one DEFAULT clause can appear on the SERIAL directive
  !$acc serial default(present) default(none)
  !$acc end serial

  !$acc serial device_type(*) async wait
  !$acc end serial

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

  !$acc kernels async
  !$acc end kernels

  !$acc kernels async(1)
  !$acc end kernels

  !$acc kernels async(async1)
  !$acc end kernels

  !$acc kernels wait(wait1)
  !$acc end kernels

  !$acc kernels wait(wait1, wait2)
  !$acc end kernels

  !$acc kernels wait(1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(queues: 1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(1) wait(2) async(3)
  !$acc end kernels

  !$acc kernels wait(devnum: 1: 1, 2) async(3)
  !$acc end kernels

  !$acc kernels wait(devnum: 1: queues: 1, 2) async(3)
  !$acc end kernels

  !$acc kernels num_gangs(8)
  !$acc end kernels

  !$acc kernels num_workers(8)
  !$acc end kernels

  !$acc kernels vector_length(128)
  !$acc end kernels

  !$acc kernels if(.true.)
  !$acc end kernels

  !$acc kernels if(ifCondition)
  !$acc end kernels

  !ERROR: At most one IF clause can appear on the KERNELS directive
  !$acc kernels if(.true.) if(ifCondition)
  !$acc end kernels

  !$acc kernels self
  !$acc end kernels

  !$acc kernels self(.true.)
  !$acc end kernels

  !$acc kernels self(ifCondition)
  !$acc end kernels

  !$acc kernels copy(aa) copyin(bb) copyout(cc)
  !$acc end kernels

  !$acc kernels copy(aa, bb) copyout(zero: cc)
  !$acc end kernels

  !$acc kernels present(aa, bb) create(cc)
  !$acc end kernels

  !$acc kernels copyin(readonly: aa, bb) create(zero: cc)
  !$acc end kernels

  !$acc kernels deviceptr(aa, bb) no_create(cc)
  !$acc end kernels

  !$acc kernels attach(aa, bb, cc)
  !$acc end kernels

  !ERROR: PRIVATE clause is not allowed on the KERNELS directive
  !$acc kernels private(aa, bb, cc)
  !$acc end kernels

  !$acc kernels default(none)
  !$acc end kernels

  !$acc kernels default(present)
  !$acc end kernels

  !ERROR: At most one DEFAULT clause can appear on the KERNELS directive
  !$acc kernels default(none) default(present)
  !$acc end kernels

  !$acc kernels device_type(*)
  !$acc end kernels

  !$acc kernels device_type(1)
  !$acc end kernels

  !$acc kernels device_type(1, 3)
  !$acc end kernels

  !$acc kernels device_type(*) async wait num_gangs(8) num_workers(8) vector_length(128)
  !$acc end kernels

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

  !$acc wait

  !$acc wait async

  !$acc wait(1)
  !$acc wait(1, 2)

  !$acc wait(queues: 1)
  !$acc wait(queues: 1, 2)

  !$acc wait(devnum: 1: 3)
  !$acc wait(devnum: 1: 3, 4)

  !$acc wait(devnum: 1: queues: 3)
  !$acc wait(devnum: 1: queues: 3, 4)

  !$acc wait(1) if(.true.)

  !ERROR: At most one IF clause can appear on the WAIT directive
  !$acc wait(1) if(.true.) if(.false.)

  !$acc wait(1) if(.true.) async

  !$acc wait(1) if(.true.) async(1)

  !ERROR: At most one ASYNC clause can appear on the WAIT directive
  !$acc wait(1) if(.true.) async(1) async

  !$acc parallel
  !$acc atomic update
  c(i) = c(i) + 1

  !$acc atomic update
  c(i) = c(i) + 1
  !$acc end atomic

  !$acc atomic write
  c(i) = 10

  !$acc atomic write
  c(i) = 10
  !$acc end atomic

  !$acc atomic read
  i = c(i)

  !$acc atomic read
  i = c(i)
  !$acc end atomic

  !$acc atomic capture
  c(i) = i
  i = i + 1
  !$acc end atomic
  !$acc end parallel
  t%arr(i) = 2.0

  !$acc cache(a(i))
  !$acc cache(a(1:2,3:4))
  !$acc cache(a)
  !$acc cache(readonly: a, aa)
  !$acc cache(readonly: a(i), aa(i, i))
  !$acc cache(t%arr)
  !$acc cache(ta(1:2)%arr)
  !$acc cache(ta(1:2)%arr(1:4))

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(ta(1:2)%s)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(i)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(t%s)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(/i/)

end program openacc_clause_validity
