!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


!dummy argument
module typemodule
  type testtype
    integer::intnum=1
    real::realnum=1.2
  end type testtype
end module


subroutine nlme(myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr)
  use typemodule
  real,pointer::myptr
  real,pointer::myptra(:,:)
  real,pointer::myptrs(:,:)
  real,allocatable::fromss
  real,allocatable::fromaa(:)
  character(:),allocatable::allo_char
  type(testtype),pointer::dtptr

  namelist/mygroup2/myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr

  open(12, file='namelist2.out' , action='write', delim='APOSTROPHE')
  write(12, nml=mygroup2)
  close(12)

end 


program testme

  use typemodule
  parameter(N=37)
  real rk
  integer i,j,k
  logical expect(N)
  logical result(N)

  real,pointer::myptr
  real::myptr_r
  real,target::mytarget

  real,pointer::myptra(:,:)
  real,pointer::myptrs(:,:)
  real,target::mytargeta(1:5,1:5)
  real::myptra_r(1:5,1:5)
  real::myptrs_r(1:3,1:3)

  real,allocatable::fromss
  real::fromss_r
  real,allocatable::fromaa(:)
  real::fromaa_r(3)

  character(:),allocatable::allo_char
  character*2::aaaa='dh'
  character*1200 result1
  character*1200 result2

  type(testtype),target::dttarget
  type(testtype),pointer::dtptr
  type(testtype)::dtt

  interface
subroutine nlme(myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr)
  use typemodule
  real,pointer::myptr
  real,pointer::myptra(:,:)
  real,pointer::myptrs(:,:)
  real,allocatable::fromss
  real,allocatable::fromaa(:)
  character(:),allocatable::allo_char
  type(testtype),pointer::dtptr
end subroutine
  end interface

  namelist/mygroup/myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr
  namelist/mygroup2/myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr

! pointer scalar
  myptr=>mytarget
  mytarget=2.0
  myptr_r = mytarget

! allocatable scalar
  allocate(fromss)
  fromss=5.3
  fromss_r=fromss

! pointer array
  myptra=>mytargeta
  rk=1.0
  do i = 1, 5
    do j = 1, 5
        mytargeta(j,i) =  rk
        rk = rk + .1
    end do
  end do
  myptra_r=mytargeta

! alloctable array
  allocate(fromaa(3))
  fromaa=1.3
  fromaa_r=1.3

! deferchar
  allocate(character(2)::allo_char)
  allo_char='dh'

!derived type
   dtptr=>dttarget

!section array
  myptrs=>mytargeta(2:4, 1:3)
  myptrs_r=mytargeta(2:4,1:3)
!  myptrs_r = myptrs

 
! write
  open(10, file='namelist.out' , action='write', DELIM='APOSTROPHE')
  write(10, nml=mygroup)
  close(10)

! read
  open(11, file='namelist.out', action='read')
  read(11, nml=mygroup)
  close(11)

! dummy argument
  call nlme(myptr,myptra,myptrs,fromss,fromaa,allo_char,dtptr)


  open(13, file='namelist2.out', action='read', DELIM='APOSTROPHE')
  read(13, nml=mygroup2)
  close(13)
  write (*, nml=mygroup)


! do a diff on file
  open(11, file='namelist.out', action='read')
  open(12, file='namelist2.out', action='read')

  read(11, '(A)') result1
  read(12, '(A)') result2
  close(11)
  close(12)

  result(1) = result1 .eq. result2

  call check(result, expect, N)

end
