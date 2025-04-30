!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for call function take quad complex argument.

program test
    implicit none
    integer, parameter :: n = 6
    integer :: i
    real(16), parameter :: q_tol = 5E-33_16
    complex(16) :: result(n), expect(n)
    complex(16) :: c1, c2, c3

    expect = (/ &
      (4.4_16, 6.6_16), &
      (4.4_16, 6.6_16), &
      (2.2_16, 2.2_16), &
      (13.2_16, 15.4_16), &
      (13.2_16, 5.5_16), &
      (1.1_16, 1.1_16) &
    /)

    c1 = (1.1_16, 2.2_16)
    c2 = (3.3_16, 4.4_16)
    result(1) = subs1(c1, c2)
    result(2) = subs2(c1, c2)
    call subs3(c1, c2, c3)
    result(3) = c3
    result(4) = subs1((5.5_16, 6.6_16), (7.7_16, 8.8_16))

    result(5) = subs2((9.9_16, 1.1_16), c2)

    call subs3(c1, (2.2_16, 3.3_16), result(6))

    do i = 1, n
      if (expect(i)%re .eq. 0.0_16) then
        if (result(i)%re .ne. expect(i)%re) STOP i
      else
        if (abs((result(i)%re - expect(i)%re) / expect(i)%re) .gt. q_tol) STOP i
      endif
      if (expect(i)%im .eq. 0.0_16) then
        if (result(i)%im .ne. expect(i)%im) STOP i
      else
        if (abs((result(i)%im - expect(i)%im) / expect(i)%im) .gt. q_tol) STOP i
      endif
    enddo
    print *, 'PASS'

  contains
  function  subs1(a, b ) result(c)
    complex(16), intent(in) :: a, b
    complex(16) :: c
    c = a + b
  end function

  complex(16) function  subs2(a, b)
  complex(16), intent(in) :: a, b
    subs2 = a - -b
  end function

  subroutine subs3(a, b, c)
    complex(16), intent(in) ::  a, b
    complex(16), intent(out) ::  c
    c = -a + b
  end subroutine

end program
