! RUN: %python %S/test_errors.py %s %flang_fc1
! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!
!Section 11.1.7.4.3, paragraph 2 states:
!  Except for the incrementation of the DO variable that occurs in step (3), 
!  the DO variable shall neither be redefined nor become undefined while the
!  DO construct is active.

subroutine s1()

  ! Redefinition via intrinsic assignment (section 19.6.5, case (1))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    ivar = 99
  end do

  ! Redefinition in the presence of a construct association
  associate (avar => ivar)
    do ivar = 1,20
      print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
      avar = 99
    end do
  end associate

  ivar = 99

  ! Redefinition via intrinsic assignment (section 19.6.5, case (1))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    ivar = 99
  end do

  ivar = 99

end subroutine s1

subroutine s2()

  integer :: ivar

  read '(I10)', ivar

  ! Redefinition via an input statement (section 19.6.5, case (3))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read '(I10)', ivar
  end do

  ! Redefinition via an input statement (section 19.6.5, case (3))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read '(I10)', ivar
  end do

end subroutine s2

subroutine s3()

  integer :: ivar

  ! Redefinition via use as a DO variable (section 19.6.5, case (4))
  do ivar = 1,10
!ERROR: Cannot redefine DO variable 'ivar'
    do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
      do ivar = 1,30
        print *, "hello"
      end do
    end do
  end do

  ! This one's OK, even though we used ivar previously as a DO variable
  ! since it's not a redefinition
  do ivar = 1,40
    print *, "hello"
  end do

  ! Redefinition via use as a DO variable (section 19.6.5, case (4))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    do ivar = 1,20
      print *, "hello"
    end do
  end do

end subroutine s3

subroutine s4()

  integer :: ivar
  real :: x(10)

  print '(f10.5)', (x(ivar), ivar = 1, 10)

  ! Redefinition via use as a DO variable (section 19.6.5, case (5))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    print '(f10.5)', (x(ivar), ivar = 1, 10)
  end do

  ! Redefinition via use as a DO variable (section 19.6.5, case (5))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    print '(f10.5)', (x(ivar), ivar = 1, 10)
  end do

end subroutine s4

subroutine s5()

  integer :: ivar
  real :: x

  read (3, '(f10.5)', iostat = ivar) x

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (7))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(f10.5)', iostat = ivar) x
  end do

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (7))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(f10.5)', iostat = ivar) x
  end do

end subroutine s5

subroutine s6()

  character (len=3) :: key
  integer :: chars
  integer :: ivar
  real :: x

  read (3, '(a3)', advance='no', size = chars) key

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (9))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(a3)', advance='no', size = ivar) key
    print *, "hello"
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (9))
  do concurrent (ivar = 1:10)
!ERROR: ADVANCE specifier is not allowed in DO CONCURRENT
!ERROR: Cannot redefine DO variable 'ivar'
    read (3, '(a3)', advance='no', size = ivar) key
    print *, "hello"
  end do

end subroutine s6

subroutine s7()

  integer :: iostatVar, nextrecVar, numberVar, posVar, reclVar, sizeVar

  inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
    pos=posVar, recl=reclVar, size=sizeVar)

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (10))
  do iostatVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'iostatvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in IOSTAT specifier (section 19.6.5, case (10))
  do concurrent (iostatVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'iostatvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NEXTREC specifier (section 19.6.5, case (10))
  do nextrecVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'nextrecvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NEXTREC specifier (section 19.6.5, case (10))
  do concurrent (nextrecVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'nextrecvar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NUMBER specifier (section 19.6.5, case (10))
  do numberVar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'numbervar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in NUMBER specifier (section 19.6.5, case (10))
  do concurrent (numberVar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'numbervar'
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in RECL specifier (section 19.6.5, case (10))
  do reclVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'reclvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in RECL specifier (section 19.6.5, case (10))
  do concurrent (reclVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'reclvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in POS specifier (section 19.6.5, case (10))
  do posVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'posvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in POS specifier (section 19.6.5, case (10))
  do concurrent (posVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'posvar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (10))
  do sizeVar = 1,20
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'sizevar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

  ! Redefinition via use in SIZE specifier (section 19.6.5, case (10))
  do concurrent (sizeVar = 1:10)
    print *, "hello"
    inquire(3, iostat=iostatVar, nextrec=nextrecVar, number=numberVar, & 
!ERROR: Cannot redefine DO variable 'sizevar'
      pos=posVar, recl=reclVar, size=sizeVar)
  end do

end subroutine s7

subroutine s8()

  Integer :: ivar
  integer, pointer :: ip

  allocate(ip, stat = ivar)

  ! Redefinition via a STAT= specifier (section 19.6.5, case (16))
  do ivar = 1,20
!ERROR: Cannot redefine DO variable 'ivar'
    allocate(ip, stat = ivar)
    print *, "hello"
  end do

  ! Redefinition via a STAT= specifier (section 19.6.5, case (16))
  do concurrent (ivar = 1:10)
!ERROR: Cannot redefine DO variable 'ivar'
    allocate(ip, stat = ivar)
    print *, "hello"
  end do

end subroutine s8

subroutine s9()

  Integer :: ivar

  ! OK since the DO CONCURRENT index-name exists only in the scope of the
  ! DO CONCURRENT construct
  do ivar = 1,20
    print *, "hello"
    do concurrent (ivar = 1:10)
      print *, "hello"
    end do
  end do

  ! OK since the DO CONCURRENT index-name exists only in the scope of the
  ! DO CONCURRENT construct
  do concurrent (ivar = 1:10)
    print *, "hello"
    do concurrent (ivar = 1:10)
      print *, "hello"
    end do
  end do

end subroutine s9

subroutine s10()

  Integer :: ivar
  open(file="abc", newunit=ivar)

  ! Redefinition via NEWUNIT specifier (section 19.6.5, case (29))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    open(file="abc", newunit=ivar)
  end do

  ! Redefinition via NEWUNIT specifier (section 19.6.5, case (29))
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    open(file="abc", newunit=ivar)
  end do

end subroutine s10

subroutine s11()

  Integer, allocatable :: ivar

  allocate(ivar)

  ! This look is OK
  do ivar = 1,20
    print *, "hello"
  end do

  ! Redefinition via deallocation (section 19.6.6, case (10))
  do ivar = 1,20
    print *, "hello"
!ERROR: Cannot redefine DO variable 'ivar'
    deallocate(ivar)
  end do

  ! This case is not applicable since the version of "ivar" that's inside the
  ! DO CONCURRENT has the scope of the DO CONCURRENT construct.  Within that
  ! scope, it does not have the "allocatable" attribute, so the following test
  ! fails because you can only deallocate a variable that's allocatable.
  do concurrent (ivar = 1:10)
    print *, "hello"
!ERROR: Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
    deallocate(ivar)
  end do

end subroutine s11

subroutine s12()

  Integer :: ivar, jvar

  call intentInSub(jvar, ivar)
  do ivar = 1,10
    call intentInSub(jvar, ivar)
  end do

  call intentOutSub(jvar, ivar)
  do ivar = 1,10
!ERROR: Cannot redefine DO variable 'ivar'
    call intentOutSub(jvar, ivar)
  end do

  call intentInOutSub(jvar, ivar)
  do ivar = 1,10
    !WARNING: Possible redefinition of DO variable 'ivar'
    call intentInOutSub(jvar, ivar)
  end do

contains
  subroutine intentInSub(arg1, arg2)
    integer, intent(in) :: arg1
    integer, intent(in) :: arg2
  end subroutine intentInSub

  subroutine intentOutSub(arg1, arg2)
    integer, intent(in) :: arg1
    integer, intent(out) :: arg2
  end subroutine intentOutSub

  subroutine intentInOutSub(arg1, arg2)
    integer, intent(in) :: arg1
    integer, intent(inout) :: arg2
  end subroutine intentInOutSub

end subroutine s12

subroutine s13()

  Integer :: ivar, jvar

  ! This one is OK
  do ivar = 1, 10
    jvar = intentInFunc(ivar)
  end do

  ! Error for passing a DO variable to an INTENT(OUT) dummy
  do ivar = 1, 10
!ERROR: Cannot redefine DO variable 'ivar'
    jvar = intentOutFunc(ivar)
  end do

  ! Error for passing a DO variable to an INTENT(OUT) dummy, more complex
  ! expression
  do ivar = 1, 10
!ERROR: Cannot redefine DO variable 'ivar'
    jvar = 83 + intentInFunc(intentOutFunc(ivar))
  end do

  do ivar = 1, 10
    !WARNING: Possible redefinition of DO variable 'ivar'
    jvar = intentInOutFunc(ivar)
  end do

contains
  function intentInFunc(dummyArg)
    integer, intent(in) :: dummyArg
    integer  :: intentInFunc

    intentInFunc = 343
  end function intentInFunc

  function intentOutFunc(dummyArg)
    integer, intent(out) :: dummyArg
    integer  :: intentOutFunc

    dummyArg = 216
    intentOutFunc = 343
  end function intentOutFunc

  function intentInOutFunc(dummyArg)
    integer, intent(inout) :: dummyArg
    integer  :: intentInOutFunc

    dummyArg = 216
    intentInOutFunc = 343
  end function intentInOutFunc

end subroutine s13
