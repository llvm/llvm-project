! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests F2003 defined I/O (unformatted write)
! This example is based on example from the Fortran 2003 Handbook (page 336)

module rational_stuff
  type rational
    integer n, d
    real :: rat_rslt=0.0
  end type

  interface write (unformatted)
    module procedure write_rational_value
  end interface write (unformatted)
  interface read (unformatted)
    module procedure read_rational_value
  end interface read (unformatted)


 contains
  subroutine write_rational_value(dtv, unit, iostat, iomsg)

  class(rational), intent(inout) :: dtv
  integer, intent(in) :: unit
  integer, intent(out) :: iostat
  character(len=*), intent(inout) :: iomsg
  real rat 

     if (dtv%d .ne. 0) then
       dtv%rat_rslt = real(dtv%n)/real(dtv%d)
       write(unit), dtv%n, dtv%d, dtv%rat_rslt  
     endif
  end subroutine

  subroutine read_rational_value(dtv, unit, iostat, iomsg)

  class(rational), intent(inout) :: dtv
  integer, intent(in) :: unit
  integer, intent(out) :: iostat
  character(len=*), intent(inout) :: iomsg

  end subroutine
end module

  use rational_stuff
  type(rational) x, y
  logical rslt(3), expect(3)
  x = rational(2,3)
  open(10, file='io13.output', status='replace', form='unformatted')
  write(10) , x
  close(10)
  open(10, file='io13.output', status='old', form='unformatted')
  read(10), y%n, y%d, y%rat_rslt
  close(10)

  !print *, x%n, x%d, x%rat_rslt
  !print *, y%n, y%d, y%rat_rslt

  expect = .true.
  rslt(1) = x%n .eq. y%n
  rslt(2) = x%d .eq. y%d
  rslt(3) = x%rat_rslt .eq. y%rat_rslt

  call check(rslt, expect, 3) 
  end



