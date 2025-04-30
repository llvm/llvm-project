!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!



  integer, parameter :: NBR_TSTS = 26
  complex*16 :: cplx = (1.1,2.2)
  complex*16 :: cplxarr(3) = [(1.1,2.2), (3.3, 4.4), (5.5, 6.6)]
  real*8 :: r
  type t
     complex*16 :: cplxarr(3)
  end type
  type(t):: tinst = t( [(1.1,2.2), (3.3, 4.4), (5.5, 6.6)] )
  real*8 :: expect(NBR_TSTS)
  data expect /3.14, 2.2, 3.14, -3.14, 3.14, 4.4, 3.14, -3.14,   &
               3.14, 4.4, 3.14, -3.14, 3.14, -3.14, 3.14, -3.14, &
               3.14, -3.14, 1.1, 2.2, 5.5, 6.6, 1.1, 2.2, 5.5, 6.6/

  real*8 :: result(NBR_TSTS)

  cplx%re = 3.14
  result(1) = cplx%re
  result(2) = imag(cplx)
  cplx%im = -3.14
  result(3) = real(cplx)
  result(4) = cplx%im
  cplxarr(2)%re = 3.14
  result(5) = cplxarr(2)%re
  result(6) = imag(cplxarr(2))
  cplxarr(2)%im = -3.14
  result(7) = real(cplxarr(2))
  result(8) = cplxarr(2)%im
  tinst%cplxarr(2)%re = 3.14
  result(9) = tinst%cplxarr(2)%re
  result(10) = imag(tinst%cplxarr(2))
  tinst%cplxarr(2)%im = -3.14
  result(11) = real(tinst%cplxarr(2))
  result(12) = tinst%cplxarr(2)%im

!  print *, cplxe
!  print *, cplxarr 
!  print *, tinst%cplxarr

  call pass_cmplxpart(cplx%re, 13)
  call pass_cmplxpart(cplx%im, 14)
  call pass_cmplxpart(cplxarr(2)%re, 15)
  call pass_cmplxpart(cplxarr(2)%im, 16)
  call pass_cmplxpart(tinst%cplxarr(2)%re, 17)
  call pass_cmplxpart(tinst%cplxarr(2)%im, 18)

  result(19) = real(cplxarr(1))
  result(20) = imag(cplxarr(1))
  result(21) = real(cplxarr(3))
  result(22) = imag(cplxarr(3))
  result(23) = real(tinst%cplxarr(1))
  result(24) = imag(tinst%cplxarr(1))
  result(25) = real(tinst%cplxarr(3))
  result(26) = imag(tinst%cplxarr(3))

  call checkd(result, expect, NBR_TSTS)
 contains
  subroutine pass_cmplxpart(r, tstnbr)
    real*8 :: r
    integer :: tstnbr

    result(tstnbr) = r
!   print *,"print_cmplxpart: ", r
  end subroutine
end program
