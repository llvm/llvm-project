!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module prng
    integer, save :: seed(4)
    integer :: idx = 1

 contains
  subroutine print_seed()
    print *, seed
  end subroutine

  impure elemental subroutine random(x)
      real :: x

      seed(idx) = x
      idx = idx + 1
      call print_seed()
  end subroutine

  integer impure elemental function getx(y,x)
    integer, intent(in):: y,x
    integer, volatile :: allowed
    if( x .eq. 0 ) stop 'DIVIDE BY ZERO'
    getx = y/x
  end function getx
  
  subroutine sub (zzz,www) 
    integer :: yyy(10), zzz(10), www(10)
  
    yyy = 1
    write (6,*) getx(99,zzz+yyy)
    write (6,*)
    call foo(getx(99,zzz+yyy),www)
  end subroutine sub
  
  subroutine foo(a,b)
    integer :: a(10)
    integer :: b(10)
    b = a
  end subroutine foo

end module


program p
 use prng

  integer zzz(10), rslt(14)
  integer expect(14)
  data expect /49,33,24,19,16,14,12,11,9,9,1,2,3,4/
  real s(4)
  data s /1.1, 2.2, 3.3, 4.4/
  do i=1,14
     zzz(i) = i
  enddo
  rslt = 0
  call sub(zzz,rslt)

  call random(s)

  rslt(11:14) = seed

  call check (rslt, expect, 14)
endprogram
