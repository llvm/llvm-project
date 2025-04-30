!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! test control edit descriptor dc,pp with directed-list and format statement


program testall_dc_dp
 parameter(N=57)
 character*10 charme,compareme,outchar(9)
 complex intput_c, output_c(3)
 real input_f(4), output_f(12)
 integer result(N), expect(N)
 expect =-1

! type testme
!    integer a
!    real    b
!    real    c
!    integer d
! end type testme
!type(testme)   yesatest

x=1.234
input_c=1.245
charme='\'a,bc\''
compareme='a,bc'

input_f(1)=1.3
input_f(2)=1.4
input_f(3)=1.5
input_f(4)=1.6

! directed-list
open(28, file='decimal.out', action='write', form='formatted', decimal='comma')
open(29, file='point.out', action='write', form='formatted', decimal='point')
open(30, file='default.out', action='write', form='formatted')

! write out file default
write(28,*),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4),"\n"
write(29,*),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4), "\n"
write(30,*),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4), "\n"

! write with rw03 and default
write(28,*, decimal='point'),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4)
write(29,*, decimal='comma'),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4)
write(30,*),input_f(1),input_c,charme,charme,input_f(2),input_f(3),charme,input_f(4)
close(28)
close(29)
close(30)

open(28, file='decimal.out', action='read', form='formatted', decimal='comma')
open(29, file='point.out', action='read', form='formatted', decimal='point')
open(30, file='default.out', action='read', form='formatted')


!clear output
output_f  = 0;
! read in using file default
read(28, * , decimal='comma' ),output_f(1),output_c(1),outchar(1),outchar(2),output_f(2),output_f(3),outchar(3),output_f(4)
read(29,*),output_f(5),output_c(2),outchar(4),outchar(5),output_f(6),output_f(7),outchar(6),output_f(8)
read(30,* ),output_f(9),output_c(3),outchar(7),outchar(8),output_f(10),output_f(11),outchar(9),output_f(12)

result(1) = output_f(1) == input_f(1)
result(2) = output_f(2) == input_f(2)
result(3) = output_f(3) == input_f(3)
result(4) = output_f(4) == input_f(4)
result(5) = output_f(5) == input_f(1)
result(6) = output_f(6) == input_f(2)
result(7) = output_f(7) == input_f(3)
result(8) = output_f(8) == input_f(4)
result(9) = output_f(9) == input_f(1)
result(10) = output_f(10) == input_f(2)
result(11) = output_f(11) == input_f(3)
result(12) = output_f(12) == input_f(4)
result(13) = outchar(1) == compareme
result(14) = outchar(2) == compareme
result(15) = outchar(3) == compareme
result(16) = outchar(4) == compareme
result(17) = outchar(5) == compareme
result(18) = outchar(6) == compareme
result(19) = outchar(7) == compareme
result(20) = outchar(8) == compareme
result(21) = outchar(9) == compareme
result(22) = input_c == output_c(1)
result(23) = input_c == output_c(2)
result(24) = input_c == output_c(3)

!clear output
output_f  = 0;

! read in using file with rw03
read(28,*, decimal='point'),output_f(1),output_c(1),outchar(1),outchar(2),output_f(2),output_f(3),outchar(3),output_f(4)
read(29,*, decimal='comma'),output_f(5),output_c(2),outchar(4),outchar(5),output_f(6),output_f(7),outchar(6),output_f(8)
read(30,*),output_f(9),output_c(3),outchar(7),outchar(8),output_f(10),output_f(11),outchar(9),output_f(12)

result(25) = output_f(1) == input_f(1)
result(26) = output_f(2) == input_f(2)
result(27) = output_f(3) == input_f(3)
result(28) = output_f(4) == input_f(4)
result(29) = output_f(5) == input_f(1)
result(30) = output_f(6) == input_f(2)
result(31) = output_f(7) == input_f(3)
result(32) = output_f(8) == input_f(4)
result(33) = output_f(9) == input_f(1)
result(34) = output_f(10) == input_f(2)
result(35) = output_f(11) == input_f(3)
result(36) = output_f(12) == input_f(4)
result(37) = outchar(1) == compareme
result(38) = outchar(2) == compareme
result(39) = outchar(3) == compareme
result(40) = outchar(4) == compareme
result(41) = outchar(5) == compareme
result(42) = outchar(6) == compareme
result(43) = outchar(7) == compareme
result(44) = outchar(8) == compareme
result(45) = outchar(9) == compareme
result(46) = input_c == output_c(1)
result(47) = input_c == output_c(2)
result(48) = input_c == output_c(3)


! dc,dp control descriptor
open(31, file='descriptor.out', action='write', form='formatted')
write(31,10) input_f, input_f(1)
10 format (" Default ", f4.2," Point ",f4.2," AComma ",dc, f4.2, " AComma " ,f4.2, " Point ", dp,f4.2)

close(31)

output_f = 0
open(31, file='descriptor.out', action='read', form='formatted')
read (31,11) outchar(1),output_f(1),outchar(2),output_f(2),outchar(3),output_f(3), outchar(4),output_f(4),outchar(5),output_f(5)
11 format (A9,f4.2,A7,f4.2,(A8),dc,f4.2,(A8),f4.2,A7,dp,f4.2)

result(49) = outchar(1) == " Default "
result(50) = outchar(2) == " Point "
result(51) = outchar(3) == " AComma "
result(52) = outchar(4) == " AComma "
result(53) = outchar(5) == " Point "
result(54) = output_f(1) == input_f(1)
result(55) = output_f(2) == input_f(2)
result(56) = output_f(3) == input_f(3)
result(57) = output_f(4) == input_f(4)
result(57) = output_f(5) == input_f(1)

call check(result,expect, N)

end
