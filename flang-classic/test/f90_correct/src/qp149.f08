! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : real(16) ^ real(16)

program test
  integer, parameter :: n = 16
  !integer, parameter :: m = 2 * n
  real(16), parameter :: q_tol = 5E-33_16
  integer :: i
  real(16) :: c1, czero
  real(16) :: rst1 = 1.58879788522658767867845243213578736_16 ** 666.512059852384793252030932094990168_16
  real(16) :: rst2 = 2.58879788522658767867812344523213578_16 ** -15.6912909810929581908241824108598451_16
  real(16) :: rst3 = 2.58879788522658767867845246863213578_16 ** 0.0_16
  real(16) :: rst4 = 2.58879788522658767867845653243213578_16 ** 1.0_16
  real(16) :: rst5 = 25.8879788526587678678452432138566578_16 ** 2.0_16
  real(16) :: rst6 = 0.0_16 ** 0.0_16
  real(16) :: rst7 = 0.0_16 ** 1.0_16
  real(16) :: rst8 = 0.0_16 ** 2.0_16

  real(16), parameter :: rst9 = -156.58879788522658767867845243213578_16 ** 77.777777777777777777777777777777777_16
  real(16), parameter :: rst10 = -223.58879788522658767878845243213578_16 ** -17.777777777777777777777777777777777_16
  real(16), parameter :: rst11 = 1.58879788522658767867845243213578_16 ** 0.0_16
  real(16), parameter :: rst12 = 1.58879788522658767867845243213578_16 ** 1.0_16
  real(16), parameter :: rst13 = 1.58879788522658767867845243213578_16 ** 2.0_16
  real(16), parameter :: rst14 = 0.0_16 ** 0.0_16
  real(16), parameter :: rst15 = 0.0_16 ** 1.0_16
  real(16), parameter :: rst16 = 0.0_16 ** 2.0_16
  real(16) :: result(n), expect(n)

  expect(1) = 1.03438481897629616059821308280450412E+0134_16
  expect(2) = 3.29576927393598767988936282545539133E-0007_16
  expect(3) = 1.00000000000000000000000000000000000_16
  expect(4) = 2.58879788522658767867845653243213560_16
  expect(5) = 670.187449075707575166743549361940742_16
  expect(6) = 1.00000000000000000000000000000000000_16
  expect(7) = 0.00000000000000000000000000000000000_16
  expect(8) = 0.00000000000000000000000000000000000_16
  expect(9) = -5.05369734583473745208452917797946642E+0170_16
  expect(10) = -1.70607952618982964551153874742749618E-0042_16
  expect(11) = 1.00000000000000000000000000000000000_16
  expect(12) = 1.58879788522658767867845243213578004_16
  expect(13) = 2.52427872010047727435411161462975619_16
  expect(14) = 1.00000000000000000000000000000000000_16
  expect(15) = 0.00000000000000000000000000000000000_16
  expect(16) = 0.00000000000000000000000000000000000_16

  result(1) = rst1
  result(2) = rst2
  result(3) = rst3
  result(4) = rst4
  result(5) = rst5
  result(6) = rst6
  result(7) = rst7
  result(8) = rst8
  result(9) = rst9
  result(10) = rst10
  result(11) = rst11
  result(12) = rst12
  result(13) = rst13
  result(14) = rst14
  result(15) = rst15
  result(16) = rst16

  do i = 1, n
    if(expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo

  print *, 'PASS'
end
