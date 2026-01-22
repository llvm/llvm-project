! RUN: bbc -fopenacc -emit-hlfir -lower-do-while-to-scf-while %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPacc_nested_loops()
! CHECK: scf.while
subroutine acc_nested_loops()
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none

  integer, parameter :: loopcount = 1000
  real(real64), parameter :: precision = 1.0e-3_real64

  real(real64), allocatable :: a(:,:)
  real(real64) :: avg
  integer :: x, y

  allocate(a(10, loopcount))

  ! Initialize input data
  do x = 1, 10
    do y = 1, loopcount
      a(x,y) = 1.0_real64
    end do
  end do

  !$acc data copy(a(1:10, 1:loopcount))
  !$acc parallel
  !$acc loop
  do x = 1, 10
    avg = 0.0_real64
    do while (avg - 1000.0_real64 < precision * real(loopcount, real64))
      avg = 0.0_real64

      !$acc loop
      do y = 1, loopcount
        a(x, y) = a(x, y) * 1.5_real64
      end do

      !$acc loop reduction(+:avg)
      do y = 1, loopcount
        avg = avg + (a(x, y) / real(loopcount, real64))
      end do
    end do
  end do
  !$acc end parallel
  !$acc end data

  deallocate(a)
end subroutine acc_nested_loops

