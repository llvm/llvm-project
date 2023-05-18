! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.5.3 Kernels

program openacc_kernels_validity

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

  !ERROR: Argument `aa` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc kernels attach(dd, p, aa)
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

  !$acc kernels device_type(default)
  !$acc end kernels

  !$acc kernels device_type(default, host)
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

  do i = 1, 100
    !$acc kernels
    !ERROR: CYCLE to construct outside of KERNELS construct is not allowed
    if (i == 10) cycle
    !$acc end kernels
  end do

  !$acc kernels
  do i = 1, 100
    if (i == 10) cycle
  end do
  !$acc end kernels

end program openacc_kernels_validity
