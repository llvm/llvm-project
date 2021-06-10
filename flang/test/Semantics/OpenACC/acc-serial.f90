! RUN: %S/../test_errors.sh %s %t %flang -fopenacc
! REQUIRES: shell

! Check OpenACC clause validity for the following construct and directive:
!   2.5.2 Serial

program openacc_serial_validity

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
  real(8), dimension(:), allocatable :: dd
  real(8), pointer :: p
  logical :: ifCondition = .TRUE.
  type(atype) :: t
  type(atype), dimension(10) :: ta

  real(8), dimension(N) :: a, f, g, h

  !$acc serial
  !ERROR: Directive SET may not be called within a compute region
  !$acc set default_async(i)
  !$acc end serial

  !$acc serial
  !$acc loop
  do i = 1, N
    !ERROR: Directive SET may not be called within a compute region
    !$acc set default_async(i)
    a(i) = 3.14
  end do
  !$acc end serial

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

  !ERROR: Argument `aa` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc serial attach(aa, dd, p)
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

end program openacc_serial_validity
