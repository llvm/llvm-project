!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test f2008 FINDLOC intrinsic

program p 
 use check_mod

    integer, parameter :: N=69
    integer :: rslts(N)
    integer :: expct(N) = (/ &
    3, &	! findloc(arr_1d1,42):  
    3, &	! findloc(arr_1d1,42,1):  
    0, &	! findloc(arr_1d1,42,1,l_1arr_1d1,42,1,l_1d):  
    3, &	! findloc(arr_1d1,1,l_1d2):  
    3, &	! findloc(arr_1d1,42, back=.true.):  
    3, &	! findloc(arr_1d2,42,back=.true.):  
    2, &	! findloc(arr_1d2,42,l_1d,back=.true.):  
    3, &	! findloc(arr_1d3,42,1,back=.true.):  
    4, &	! findloc(arr_1d4,1,l_1d,back=.true.):  
    0, &	! findloc(arr_1d1,1,l_1d,back=.true.):  
    2, 1, &	! findloc(arr_2d4,42):  
    0, 1, 0, &	! findloc(arr_2d4,42,2):  
    2, 2, 2, &	! findloc(arr_2d4,42,1):  
    0, 0, 0, &	! findloc(arr_2d1,42,2,l_2d):  
    1, 0, 0, &	! findloc(arr_2d2,42,2,l_2d):  
    0, 0, 2, &	! findloc(arr_2d3,42,2,l_2d):  
    2, 3, &	! findloc(arr_2d4,42, back=.true.):  
    2, 3, &	! findloc(arr_2d5,42,mask=l_2d, back=.true.):  
    3, 1, &	! findloc(arr_2d5,42,mask=l_2d2, back=.true.):  
    3, 1, 2, &	! findloc(arr_2d5,42,1,back=.true.):  
    2, 3, 1, &	! findloc(arr_2d5,2,42,back=.true.):  
    1, &	! findloc(a,op):  
    5, &	! findloc(a,op,c_1d):  
    1, &	! findloc(a,op,1):  
    5, &	! findloc(a,op,1,c_1d):  
    5, &	! findloc(a,op,back=.true.):  
    4, &	! findloc(a,op,c_1d2,back=.true.)):  
    0, 0, &	! findloc(a2,ayz):  
    1, 1, &	! findloc(a2,op):  
    2, 2, &	! findloc(a2,abc):  
    3, 2, &	! findloc(a2,op,c_2d):  
    0, 0, 0, &	! findloc(a2,ayz, 1):  
    0, 1, 0, &	! findloc(a2,abcd, 1):  
    2, 0, 0, &	! findloc(a2,abcd, 2):  
    3, 2, &	! findloc(a2,op,c_2d):  
    0, 3, 3, &	! findloc(a2,op,1,c_2d):  
    3, 3 &	! findloc(a2,op, c_2d2, back=.true.):    
    /)
  
    integer :: arr_1d1(4) 
    data arr_1d1/1,2,42,4/

    integer :: arr_1d2(4) 
    data arr_1d2/1,42,42,4/

    integer :: arr_1d3(4) 
    data arr_1d3/42,2,42,4/

    integer :: arr_1d4(4) 
    data arr_1d4/1,2,42,42/

    logical :: l_1d(4) = (/ .true., .true., .false., .true. /)
    logical :: l_1d2(4) = (/ .true., .false., .true., .true. /)

    integer :: arr_2d1(3,3)
    data arr_2d1/1,2,42,4,42,6,42,8,9/

    integer :: arr_2d2(3,3)
    data arr_2d2/42,2,3,42,5,6,42,8,9/

    integer :: arr_2d3(3,3)
    data arr_2d3/1,2,42,4,5,42,7,8,42/

    integer :: arr_2d4(3,3)
    data arr_2d4/1,42,3,4,42,6,7,42,9/

    integer :: arr_2d5(3,3)
    data arr_2d5/1,42,42,42,6,8,7,42,9/

    logical :: l_2d(3,3)  
    data l_2d/.true., .true., .false., .true., .false., .true., .false., .true., .true./

    logical :: l_2d2(3,3)  
    data l_2d2/.true., .true., .true., .false., .true., .true., .true., .false., .true./

    integer :: rslt
    integer :: rslt1(1)
    integer :: rslt2(2)
    integer :: rslt3(3)
    character(len=100) :: str
    character :: f

    character*6, parameter :: def = "def"
    character*6, parameter :: abc = "abc"
    character*6, parameter :: abcd = "abcd"
    character*6, parameter :: op = "op"
    character*6, parameter :: ij = "ij"
    character*6, parameter :: qr = "qr"
    character*6, parameter :: uv = "uv"

    character*6 a(5)
    data a/ op, uv, ij, op, op/
    logical :: c_1d(5) = (/ .false., .true., .true., .false., .true. /)
    logical :: c_1d2(5) = (/ .true., .true., .true., .true., .false. /)

    character*6 a2(3,3)
    data a2/ op, uv, ij, abcd, abc, op, qr,def, op/
    logical :: c_2d(3,3)
    data c_2d/ .false., .true., .true., .true.,.true.,.true.,.true.,.true.,.true./
    logical :: c_2d2(3,3)
    data c_2d2/ .true., .true., .true., .true.,.true.,.false.,.true.,.true.,.true./
    

    rslt1 = findloc(arr_1d1,42)
    rslts(1) = rslt1(1)
    !print *,"findloc(arr_1d1,42):", rslt1

    rslt = findloc(arr_1d1,42, 1)
    rslts(2) = rslt1(1)
    !print *,"findloc(arr_1d1,42,1):", rslt

    rslt = findloc(arr_1d1,42,1,l_1d)
    rslts(3) = rslt
    !print *,"findloc(arr_1d1,42,1,l_1arr_1d1,42,1,l_1d):", rslt

    rslt = findloc(arr_1d1,42,1,l_1d2)
    rslts(4) = rslt1(1)
    !print *,"findloc(arr_1d1,1,l_1d2):", rslt


    rslt1 = findloc(arr_1d1,42)
    rslts(5) = rslt1(1)
    !print *,"findloc(arr_1d1,42, back=.true.):", rslt1

    rslt1 = findloc(arr_1d2,42, back=.true.)
    rslts(6) = rslt1(1)
    !print *,"findloc(arr_1d2,42,back=.true.):", rslt1

    rslt1 = findloc(arr_1d2,42, l_1d, back=.true.)
    rslts(7) = rslt1(1)
    !print *,"findloc(arr_1d2,42,l_1d,back=.true.):", rslt1

    rslt = findloc(arr_1d3,42, 1, back=.true.)
    rslts(8) = rslt
    !print *,"findloc(arr_1d3,42,1,back=.true.):", rslt

    rslt = findloc(arr_1d4,42,1,l_1d, back=.true.)
    rslts(9) = rslt
    !print *,"findloc(arr_1d4,1,l_1d,back=.true.):", rslt

    rslt = findloc(arr_1d1,42,1,l_1d, back=.true.)
    rslts(10) = rslt
    !print *,"findloc(arr_1d1,1,l_1d,back=.true.):", rslt

    rslt2 = findloc(arr_2d4,42)   
    rslts(11:12) = rslt2(1:2)
    !print *,"findloc(arr_2d4,42):", rslt2   

    rslt3 = findloc(arr_2d4,42,2)
    rslts(13:15) = rslt3
    !print *,"findloc(arr_2d4,42,2):", rslt3   

    rslt3 = findloc(arr_2d4,42,1)
    rslts(16:18) = rslt3
    !print *,"findloc(arr_2d4,42,1):", rslt3   

    rslt3 = findloc(arr_2d1,42,2,l_2d)
    rslts(19:21) = rslt3
    !print *,"findloc(arr_2d1,42,2,l_2d):", rslt3   

    rslt3 = findloc(arr_2d2,42,2,l_2d)
    rslts(22:24) = rslt3
    !print *,"findloc(arr_2d2,42,2,l_2d):", rslt3   

    rslt3 = findloc(arr_2d3,42,2,l_2d)
    rslts(25:27) = rslt3
    !print *,"findloc(arr_2d3,42,2,l_2d):", rslt3   

    rslt2 = findloc(arr_2d4,42, back=.true.)
    rslts(28:29) = rslt2(1:2)
    !print *,"findloc(arr_2d4,42, back=.true.):", rslt2   

    rslt2 = findloc(arr_2d5,42,mask=l_2d, back=.true.)
    rslts(30:31) = rslt2(1:2)
    !print *,"findloc(arr_2d5,42,mask=l_2d, back=.true.):", rslt2   

    rslt2 = findloc(arr_2d5,42,mask=l_2d2, back=.true.)
    rslts(32:33) = rslt2(1:2)
    !print *,"findloc(arr_2d5,42,mask=l_2d2, back=.true.):", rslt2   

    rslt3 = findloc(arr_2d5,42,1,back=.true.)   
    rslts(34:36) = rslt3(1:3)
    !print *,"findloc(arr_2d5,42,1,back=.true.):", rslt3   

    rslt3 = findloc(arr_2d5,42,2,back=.true.)   
    rslts(37:39) = rslt3(1:3)
!    print *,"findloc(arr_2d5,2,42,back=.true.):", rslt3   

    rslt1 = findloc(a,op)
    rslts(40) = rslt1(1)
    !print *,"findloc(a,op):", rslt1

    rslt1 = findloc(a,op,c_1d)
    rslts(41) = rslt1(1)
    !print *,"findloc(a,op,c_1d):", rslt1

    rslt1 = findloc(a,op,1)
    rslts(42) = rslt1(1)
    !print *,"findloc(a,op,1):", rslt1

    rslt = findloc(a,op,1,c_1d)
    rslts(43) = rslt
    !print *,"findloc(a,op,1,c_1d):", rslt

    rslt1 = findloc(a,op,back=.true.)
    rslts(44) = rslt1(1)
    !print *,"findloc(a,op,back=.true.):", rslt1

    rslt1 = findloc(a,op,c_1d2,back=.true.)
    rslts(45) = rslt1(1)
    !print *,"findloc(a,op,c_1d2,back=.true.)):", rslt1

    rslt2 = findloc(a2,"ayz")
    rslts(46:47) = rslt2(1:2)
    !print *,"findloc(a2,ayz):", rslt2

    rslt2 = findloc(a2,op)
    rslts(48:49) = rslt2(1:2)
    !print *,"findloc(a2,op):", rslt2

    rslt2 = findloc(a2,abc)
    rslts(50:51) = rslt2(1:2)
    !print *,"findloc(a2,abc):", rslt2

    rslt2 = findloc(a2,op,c_2d)
    rslts(52:53) = rslt2(1:2)
    !print *,"findloc(a2,op,c_2d):", rslt2

    rslt3 = findloc(a2,"ayz", 1)
    rslts(54:56) = rslt3
    !print *,"findloc(a2,ayz, 1):", rslt3

    rslt3 = findloc(a2,abcd, 1)
    rslts(57:59) = rslt3
    !print *,"findloc(a2,abcd, 1):", rslt3

    rslt3 = findloc(a2,"abcd", 2)
    rslts(60:62) = rslt3
    !print *,"findloc(a2,abcd, 2):", rslt3

    rslt2 = findloc(a2,op,c_2d)
    rslts(63:64) = rslt2
    !print *,"findloc(a2,op,c_2d):", rslt2

    rslt3 = findloc(a2,op,1,c_2d)
    rslts(65:67) = rslt3
    !print *,"findloc(a2,op,1,c_2d):", rslt3

    rslt2 = findloc(a2,op, c_2d2, back=.true.)
    rslts(68:69) = rslt2(1:2)
    !print *,"findloc(a2,op, c_2d2, back=.true.):", rslt2

    call checki4( rslts, expct, N)

end program p
