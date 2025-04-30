! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for intrinsic sin take quad precision argument.

program test
  implicit none
  integer, parameter :: n = 16
  integer, parameter :: m = n * 4
  integer, parameter :: k = 16
  real(k), parameter :: pi_q = 3.1415926535897932384626433832795028841_k
  real(k), parameter :: q_tol = 5e-33_k
  real(k), parameter :: radians(n) = (/ &
    0.0_k,                                     & 
    0.523598775598298873077107230546583832_k,  & !1/6 pi
    0.785398163397448309615660845819875699_k,  & !1/4 pi
    1.04719755119659774615421446109316766_k,   & !1/3 pi
    1.57079632679489661923132169163975140_k,   & !1/2 pi
    2.09439510239319549230842892218633533_k,   & !2/3 pi
    2.35619449019234492884698253745962710_k,   & !3/4 pi
    2.61799387799149436538553615273291925_k,   & !5/6 pi
    3.14159265358979323846264338327950280_k,   & !pi
    3.66519142918809211153975061382608673_k,   & !7/6 pi
    3.92699081698724154807830422909937850_k,   & !5/4 pi
    4.18879020478639098461685784437267065_k,   & !4/3 pi
    4.71238898038468985769396507491925420_k,   & !3/2 pi
    5.23598775598298873077107230546583851_k,   & !5/3 pi
    5.49778714378213816730962592073912990_k,   & !7/4 pi
    5.75958653158128760384817953601242205_k    & !11/6 pi
  /)  
  integer :: i
  real(k), dimension(n) :: arg
  real(k) :: rst(n) = sin(radians)
  real(k), parameter :: rst_p(n) = sin(radians)
  real(k) :: result(m), expect(m)
  
  expect(1:n) = (/ &
    0.00000000000000000000000000000000000_k        ,&
    0.500000000000000000000000000000000000_k       ,& 
    0.707106781186547524400844362104848992_k       ,& 
    0.866025403784438646763723170752936161_k       ,& 
    1.00000000000000000000000000000000000_k        ,& 
    0.866025403784438646763723170752936161_k       ,& 
    0.707106781186547524400844362104849088_k       ,& 
    0.499999999999999999999999999999999856_k       ,& 
    8.67181013012378102479704402604335225E-0035_k  ,&
    -0.500000000000000000000000000000000000_k      ,& 
    -0.707106781186547524400844362104848992_k      ,& 
    -0.866025403784438646763723170752936257_k      ,& 
    -1.00000000000000000000000000000000000_k       ,&
    -0.866025403784438646763723170752935968_k      ,& 
    -0.707106781186547524400844362104849185_k      ,& 
    -0.499999999999999999999999999999999904_k       & 
  /)

  expect(n+1:2*n) = expect(1:n)
  expect(2*n+1:3*n) = expect(1:n)
  expect(3*n+1:4*n) = expect(1:n)
 
  arg = radians
  result(1:n) = sin(arg)
  result(n+1:2*n) = sin(radians)
  result(2*n+1:3*n) = rst
  result(3*n+1:4*n) = rst_p

  do i = 1, m
    if (expect(i) .eq. 0.0_k) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo  
 
  print *, 'PASS'

end 
