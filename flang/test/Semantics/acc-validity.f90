! RUN: %S/test_errors.sh %s %t %f18 -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.6.5 Data
!   2.5.1 Parallel
!   2.5.2 Kernels
!   2.5.3 Serial
!   2.15.1 Routine
!   2.11 Parallel Loop
!   2.11 Kernels Loop
!   2.11 Serial Loop

program openacc_clause_validity

  implicit none

  integer :: i, j
  integer :: N = 256

  !$acc declare
  real(8) :: a(256)

  !$acc enter data

  !$acc enter data copyin(zero: i)

  !$acc enter data create(readonly: i)

  !$acc data copyout(readonly: i)
  !$acc end data

  !$acc enter data copyin(i) copyout(i)

  !$acc data copy(i) if(.true.) if(.true.)
  !$acc end data

  !$acc exit data

  !$acc host_data
  !$acc end host_data

  !$acc set

  !$acc data
  !$acc end data

  !$acc data copyin(i)
  !$acc end data

  !$acc data copyin(i)

  !$acc end parallel

  !$acc update device(i) device_type(*) async


  !$acc update device(i) device_type(*) if(.TRUE.)

  !$acc parallel

  !$acc loop seq independent
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel device_type(*) num_gangs(2)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel

  !$acc loop collapse(-1)
  do i = 1, N
    do j = 1, N
      a(i) = 3.14 + j
    end do
  end do
  !$acc end parallel

  !$acc parallel

  !$acc loop device_type(*) private(i)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel

  !$acc loop gang seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel


  !$acc parallel device_type(*) if(.TRUE.)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel


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


  !$acc kernels device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels


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


  !$acc serial device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial


  !$acc serial loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

 contains

   subroutine sub1(a)
     real :: a(:)

     !$acc routine
   end subroutine sub1

   subroutine sub2(a)
     real :: a(:)

     !$acc routine seq device_type(*) nohost
   end subroutine sub2

end program openacc_clause_validity