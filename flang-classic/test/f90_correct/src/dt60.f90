!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

MODULE extendedtype
      TYPE :: TA
        real :: reala = 7.7
      END TYPE TA

      TYPE, EXTENDS(TA) :: TB
        REAL :: realb
      END TYPE TB

      TYPE, EXTENDS(TB) :: TC
        REAL :: realc
      END TYPE TC

      TYPE, EXTENDS(TC) :: TD
        REAL :: reald=3.3
      END TYPE TD


TYPE(TD) :: one=TD(1.0,2.0,3.0,4.0)
TYPE(TD) :: two=TD(reald=4.0,realb=2.0,reala=1.0,realc=3.0)
TYPE(TD) :: one1=TD(1.0,2.0,3.0)   ! this one is wrong
TYPE(TD) :: two2=TD(4.0,reald=2.0,realb=1.0,realc=3.0)
TYPE(TD) :: three=TD(realc=2.0,realb=3.0)
TYPE(TD) :: four=TD(TA(1.0),2.0,3.0)
TYPE(TD) :: five=TD(TB(TA(1.0),3.0),3.5)
TYPE(TA),parameter :: mea=TA(5.0)
TYPE(TD) :: six=TD(mea,realc=2.0,realb=3.0)
TYPE(TD) :: seven=TD(TB(realb=2.0,reala=1.0),3.0)


contains
   subroutine printme
       print *, "one:",one
       print *, "two:",two
       print *, "one1:",one1
       print *, "two2:",two2
       print *, "three:",three
       print *, "four:",four
       print *, "five:",five
       print *, "six:",six
       print *, "seven:",seven
   end subroutine printme


END MODULE extendedtype

PROGRAM test_fortran2003
  USE extendedtype
  parameter(N=36)
  real expect(N)
  data expect /1.0,2.0,3.0,4.0, &!one
                   &1.0,2.0,3.0,4.0, &!two
                   &1.0,2.0,3.0,3.3, &!one1
                   &4.0,1.0,3.0,2.0, &!two2
                   &7.7,3.0,2.0,3.3, &!three
                   &1.0,2.0,3.0,3.3, &!four
                   &1.0,3.0,3.5,3.3, &!five
                   &5.0,3.0,2.0,3.3, &!six
                   1.0,2.0,3.0,3.3 /!seven
           
                   
  real result(N)
!  call printme()
  result(1)=one%reala
  result(2)=one%realb
  result(3)=one%realc
  result(4)=one%reald
  result(5)=two%reala
  result(6)=two%realb
  result(7)=two%realc
  result(8)=two%reald
  result(9)=one1%reala
  result(10)=one1%realb
  result(11)=one1%realc
  result(12)=one1%reald
  result(13)=two2%reala
  result(14)=two2%realb
  result(15)=two2%realc
  result(16)=two2%reald
  result(17)=three%reala
  result(18)=three%realb
  result(19)=three%realc
  result(20)=three%reald
  result(21)=four%reala
  result(22)=four%realb
  result(23)=four%realc
  result(24)=four%reald
  result(25)=five%reala
  result(26)=five%realb
  result(27)=five%realc
  result(28)=five%reald
  result(29)=six%reala
  result(30)=six%realb
  result(31)=six%realc
  result(32)=six%reald
  result(33)=seven%reala
  result(34)=seven%realb
  result(35)=seven%realc
  result(36)=seven%reald

  call check(expect,result,N)
END PROGRAM test_fortran2003
