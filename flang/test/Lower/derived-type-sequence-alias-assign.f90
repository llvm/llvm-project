! RUN: bbc -emit-fir %s -o - | FileCheck %s

program alias_seq_assign
  implicit none
  integer, parameter :: N = 2

  type t
    sequence
    integer :: e(N), f(N-1)
  end type t

  type t1
    sequence
    integer :: a
    type(t) :: b
  end type t1

  type t2
    sequence
    type(t) :: b
    integer :: a
  end type t2

  type(t1) :: c
  type(t2) :: d
  integer  :: e(2*N)

  equivalence (c, d, e)

  call init_e()
  call right_shift()
  call check_right_shift()

contains

  subroutine init_e()
    integer :: i
    e = (/ (i, i=1,2*N) /)
  end subroutine init_e

  subroutine right_shift()
    c%b = d%b
  end subroutine right_shift

  subroutine check_right_shift()
    integer :: i
    if (e(1) /= 1) stop 1
    do i = 2, 2*N
      if (e(i) /= i-1) stop 1
    end do
  end subroutine check_right_shift

end program alias_seq_assign

! CHECK-LABEL: func.func @_QQmain()
! CHECK: func.func private @_FortranAAssign
