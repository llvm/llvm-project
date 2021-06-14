! RUN: %S/../test_errors.sh %s %t %flang -fopenacc
! REQUIRES: shell

! Check OpenACC clause validity for the following construct and directive:
!   2.11 Kernels Loop

program openacc_kernels_loop_validity

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

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_gangs(8)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_gangs(gang_size)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_gangs(8)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_workers(worker_size)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_workers(8)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop vector_length(vector_size)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop vector_length(128)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_gangs(gang_size)
  do i = 1, N
    a(i) = 3.14
  end do


  !$acc kernels loop if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop if(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END SERIAL LOOP directive
  !$acc end serial loop

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the KERNELS LOOP directive
  !$acc kernels loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels loop

  !$acc kernels loop async
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop async(1)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop async(async1)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(wait1)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(wait1, wait2)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(wait1) wait(wait2)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(1, 2) async(3)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(queues: 1, 2) async(3)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(devnum: 1: 1, 2) async(3)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop wait(devnum: 1: queues: 1, 2) async(3)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_gangs(8)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop num_workers(8)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop vector_length(128)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop if(.true.)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop if(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: At most one IF clause can appear on the KERNELS LOOP directive
  !$acc kernels loop if(.true.) if(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop self
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop self(.true.)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop self(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop copy(aa) copyin(bb) copyout(cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop copy(aa, bb) copyout(zero: cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop present(aa, bb) create(cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop copyin(readonly: aa, bb) create(zero: cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop deviceptr(aa, bb) no_create(cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: Argument `aa` on the ATTACH clause must be a variable or array with the POINTER or ALLOCATABLE attribute
  !$acc kernels loop attach(aa, dd, p)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop private(aa, bb, cc)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop default(none)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop default(present)
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: At most one DEFAULT clause can appear on the KERNELS LOOP directive
  !$acc kernels loop default(none) default(present)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop device_type(*)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop device_type(1)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop device_type(1, 3)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop device_type(*) async wait num_gangs(8) num_workers(8) vector_length(128)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop device_type(*) async
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the KERNELS LOOP directive
  !$acc kernels loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do

end program openacc_kernels_loop_validity
