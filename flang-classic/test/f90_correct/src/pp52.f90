! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test F2008 simply contiguous feature

program p

  integer, parameter :: NBR_TSTS=47
  integer :: tst_nbr
  integer, parameter :: n = 4
  integer :: i, j, k, l
  integer, pointer::ptr1dim(:)
  integer, pointer::ptr2dim(:, :)
  integer, pointer::ptr3dim(:,:,:)
  integer, target :: a1d(n*n)
  integer, target :: a2d(n,n)
  integer, target :: a3d(n,n,n)
  complex, pointer:: cplxptr2d(:, :)
  complex, target :: cplx3d(n,n,n)

  integer :: shape2d(2) = [n, n]
  integer :: shape3d(3) = [n, n, n]

  type :: t1
      integer :: i
  end type
  type(t1), target :: t1_2d(n,n)
  type(t1), pointer :: t1_ptr2d(:,:)

  type :: t2
    integer :: a2d(n,n)
  end type
  type(t2), target :: t2_inst
  type(t2), target :: t2_2d(n,n)
  type(t2), pointer :: t2_ptr2d(:,:)


  integer :: a2d_to_1d(16) = [11,21,31,41,12,22, 32,42,13,23,33,43,14,24,34,44]
  integer :: a3d_to_1d(64) = [111,211,311,411,121,221,321,421,131,231,331,431, &
                            141,241,341,441,112,212,312,412,122,222,322,422, &
                            132,232,332,432,142,242,342,442,113,213,313,413, &
                            123,223,323,423,133,233,333,433,143,243,343,443, &
                            114,214,314,414,124,224,324,424,134,234,334,434, &
                            144,244,344,444]
  complex :: cplx3d_to_1d(64) = [ &
 CMPLX(111.0000,-111.0000), CMPLX(211.0000,-211.0000), CMPLX(311.0000,-311.0000) , &
 CMPLX(411.0000,-411.0000), CMPLX(121.0000,-121.0000), CMPLX(221.0000,-221.0000) , &
 CMPLX(321.0000,-321.0000), CMPLX(421.0000,-421.0000), CMPLX(131.0000,-131.0000) , &
 CMPLX(231.0000,-231.0000), CMPLX(331.0000,-331.0000), CMPLX(431.0000,-431.0000) , &
 CMPLX(141.0000,-141.0000), CMPLX(241.0000,-241.0000), CMPLX(341.0000,-341.0000) , &
 CMPLX(441.0000,-441.0000), CMPLX(112.0000,-112.0000), CMPLX(212.0000,-212.0000) , &
 CMPLX(312.0000,-312.0000), CMPLX(412.0000,-412.0000), CMPLX(122.0000,-122.0000) , &
 CMPLX(222.0000,-222.0000), CMPLX(322.0000,-322.0000), CMPLX(422.0000,-422.0000) , &
 CMPLX(132.0000,-132.0000), CMPLX(232.0000,-232.0000), CMPLX(332.0000,-332.0000) , &
 CMPLX(432.0000,-432.0000), CMPLX(142.0000,-142.0000), CMPLX(242.0000,-242.0000) , &
 CMPLX(342.0000,-342.0000), CMPLX(442.0000,-442.0000), CMPLX(113.0000,-113.0000) , &
 CMPLX(213.0000,-213.0000), CMPLX(313.0000,-313.0000), CMPLX(413.0000,-413.0000) , &
 CMPLX(123.0000,-123.0000), CMPLX(223.0000,-223.0000), CMPLX(323.0000,-323.0000) , &
 CMPLX(423.0000,-423.0000), CMPLX(133.0000,-133.0000), CMPLX(233.0000,-233.0000) , &
 CMPLX(333.0000,-333.0000), CMPLX(433.0000,-433.0000), CMPLX(143.0000,-143.0000) , &
 CMPLX(243.0000,-243.0000), CMPLX(343.0000,-343.0000), CMPLX(443.0000,-443.0000) , &
 CMPLX(114.0000,-114.0000), CMPLX(214.0000,-214.0000), CMPLX(314.0000,-314.0000) , &
 CMPLX(414.0000,-414.0000), CMPLX(124.0000,-124.0000), CMPLX(224.0000,-224.0000) , &
 CMPLX(324.0000,-324.0000), CMPLX(424.0000,-424.0000), CMPLX(134.0000,-134.0000) , &
 CMPLX(234.0000,-234.0000), CMPLX(334.0000,-334.0000), CMPLX(434.0000,-434.0000) , &
 CMPLX(144.0000,-144.0000), CMPLX(244.0000,-244.0000), CMPLX(344.0000,-344.0000) , &
 CMPLX(444.0000,-444.0000) ]

  LOGICAL ::  result(NBR_TSTS) = .FALSE.
  LOGICAL ::  expect(NBR_TSTS) = .TRUE.

  do i = 1,n*n
      a1d(i) = i
  end do

  do i = 1,n
    do j = 1,n
      a2d(i,j) = i*10 + j
      t1_2d(i,j)%i = i*10 + j
      t2_inst%a2d(i,j) = i*10 + j
      t2_2d(i,j)%a2d(i,j) = i*10 + j
    end do
  end do

  do i = 1,n
    do j = 1,n
      do k = 1,n
        a3d(i,j,k) = i*100 + j*10 + k
      end do
    end do
  end do

  do i = 1,n
    do j = 1,n
      do k = 1,n
        cplx3d(i,j,k) = CMPLX(i*100 + j*10 + k, -(i*100 + j*10 + k))
      end do
    end do
  end do

!  TEST1
    tst_nbr = 1
    ptr1dim(1:n*n) => a1d
    result(tst_nbr) = all(ptr1dim .EQ. a1d)
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:n*n) =>a1d"
!    print *, "a1d:"
!    print *, a1d
!    print *,"ptr1dim:"
!    print *,ptr1dim

!  TEST2
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = all(ptr1dim(2:4) .EQ. a1d(2:4))
!    print *,"TEST", tst_nbr
!    print *, "ptr1dim(2:4)"
!    print *, ptr1dim(2:4)

!  TEST3
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n, 1:n) =>a1d
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a1d,(/4,4/)))
!    print *,"TEST", tst_nbr

!     print *,result(tst_nbr) 		! causes compiler crash

!    print *,"ptr2dim(1:n, 1:n) =>a1d"
!    print *, "a1d:"
!    print *, a1d
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST4
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr2dim(2,3) .EQ. a1d(2 + (3-1)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(2,3)"
!    print *,ptr2dim(2,3)
    
!  TEST5
    tst_nbr = tst_nbr + 1
    ptr3dim(1:2, 1:3, 1:2) => a1d
    result(tst_nbr) = all( ptr3dim .EQ. reshape(a1d, (/2,3,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:2, 1:3, 1:2) = a1d)"
!    print *,"a1d(1:12)"
!    print *,a1d(1:12)
!    print *,"ptr3dim"
!    print *,ptr3dim

!  TEST6
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(1,1,2) .EQ. a1d(1-1 + 2*(1-1) + 2*4 -1)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1,1,2)"
!    print *,ptr3dim(1,1,2)

!  TEST7
    tst_nbr = tst_nbr + 1
!    print *,"TEST", tst_nbr
    ptr1dim(1:n-2) =>a1d(2:)
    result(tst_nbr) = all(ptr1dim .EQ. a1d(2:3))
!    print *,"ptr1dim(1:n-2) =>a1d(2:)"
!    print *, "a1d(2:):"
!    print *,a1d(2:)
!    print *, "ptr1dim:"
!    print *,ptr1dim

!  TEST8
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n-2, 1:n-2) =>a1d(2:)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a1d(2:), (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1:n-2, 1:n-2) =>a1d(2:)"
!    print *, "a1d(2:):"
!    print *,a1d(2:)
!    print *, "ptr2dim:"
!    print *,ptr2dim

!  TEST9
    tst_nbr = tst_nbr + 1
    ptr3dim(2:n, 1:n-2, 1:2) =>a1d(2:)
    result(tst_nbr) = all(ptr3dim .EQ. reshape(a1d(2:), (/3,2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2:n, 1:n-2, 1:2) =>a1d(2:)"
!    print *, "a1d(2:13):"
!    print *,a1d(2:13)
!    print *, "ptr3dim:"
!    print *,ptr3dim

!  TEST10
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(2,1,2) .EQ. a1d(2 + 3*(1-1) + 6*(2-1))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2,1,2)"
!    print *,ptr3dim(2,1,2)

!  TEST11
    tst_nbr = tst_nbr + 1
    ptr1dim(1:n) =>a1d(3:6)
    result(tst_nbr) = all(ptr1dim .EQ. a1d(3:))
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:n) =>a1d(3:6)"
!    print *, "a1d(3:6):"
!    print *,a1d(3:6)
!    print *, "ptr1dim:"
!    print *,ptr1dim

!  TEST12
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n-2, 1:n-2) =>a1d(3:6)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a1d(3:), (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1:n-2, 1:n-2) =>a1d(3:6)"
!    print *, "a1d(3:6):"
!    print *,a1d(3:6)
!    print *, "ptr2dim:"
!    print *,ptr2dim

!  TEST13
    tst_nbr = tst_nbr + 1
    ptr3dim(1:n-2, 1:n-1, 1:n-2 ) =>a1d(3:6)
    result(tst_nbr) = all(ptr3dim .EQ.  reshape(a1d(3:), (/2,3,3/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:n-2, 1:n-1, 1:n-2 ) =>a1d(3:6)"
!    print *, "a1d(3:14):"
!    print *,a1d(3:14)
!    print *, "ptr3dim:"
!    print *,ptr3dim

!  TEST14
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(2,3,1) .EQ. a1d(2 + 2 + (3-1)*2 + (1-1)*6)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2,3,1)"
!    print *,ptr3dim(2,3,1)

!  TEST15
    tst_nbr = tst_nbr + 1
    ptr1dim(2:n-1) =>a1d(2:5)
    result(tst_nbr) = all(ptr1dim .EQ. a1d(2:))
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(2:n-1) =>a1d(2:5)"
!    print *, "a1d(2:3):"
!    print *, a1d(2:3)
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST16
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr1dim(3) .EQ. a1d(3)
!    print *,"TEST", tst_nbr
!    print *, "ptr1dim(3):"
!    print *, ptr1dim(3)

!  TEST17
    tst_nbr = tst_nbr + 1
    ptr2dim(2:n-1, 2:n-1) =>a1d(2:5)
    result(tst_nbr) = all( ptr2dim .EQ. reshape(a1d(2:), (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(2:n-1, 2:n-1) =>a1d(2:5)"
!    print *, "a1d(2:5):"
!    print *, a1d(2:5)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST18
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr2dim(2,3) .EQ. a1d(2 + (3-2)*2)
!    print *,"TEST", tst_nbr
!    print *, "ptr2dim(2,3):"
!    print *, ptr2dim(2,3)

!  TEST19
    tst_nbr = tst_nbr + 1
    ptr3dim(2:n-1, 3:n, 1:3) =>a1d(2:5)
    result(tst_nbr) = all(ptr3dim .EQ. reshape(a1d(2:), (/2,2,3/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2:n-1, 1:n-2, 1:3) =>a1d(2:5)"
!    print *, "a1d(2:13):"
!    print *, a1d(2:13)
!    print *, "ptr3dim:"
!    print *, ptr3dim

!  TEST20
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(2,4,2) .EQ. a1d(2 + (4-3)*2 + (2-1)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2,4,2)"
!    print *,ptr3dim(2,4,2)

!  TEST21
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(3,3,1) .EQ. a1d(3 + (3-3)*2 + (1-1)*4)
!    print *,"TEST", tst_nbr
!    print *, "ptr3dim(3,3,1):"
!    print *, ptr3dim(3,3,1)

 !===============================================

!  TEST22
    tst_nbr = tst_nbr + 1
    ptr1dim(1:n*n) =>a2d
    result(tst_nbr) = all(reshape(ptr1dim, (/n,n/)) .EQ. a2d)
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:n*n) =>a2d"
!    print *, "a2d:"
!    print *, a2d
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST23
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr1dim(10) .EQ. a2d(10/n, mod(10,n)+1)
!    print *,"TEST", tst_nbr
!    print *, "ptr1dim(10)"
!    print *, ptr1dim(10)

!  TEST24
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n, 1:n) =>a2d
    result(tst_nbr) = all(ptr2dim .EQ. a2d)
!    print *,"TEST", tst_nbr
!    result(tst_nbr) = 
!    print *,"ptr2dim(1:n, 1:n) =>a2d"
!    print *, "a2d:"
!    print *, a2d
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST25
    tst_nbr = tst_nbr + 1
    ptr3dim(1:n-2, 1:n-2, 1:2) =>a2d
    result(tst_nbr) = all(ptr3dim .EQ. reshape( (/11,21,31,41,12,22,32,42/), (/2,2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:n-2, 1:n-2, 1:2) =>a2d"
!    print *, "a2d"
!    print *, a2d
!    print *, "ptr3dim:"
!    print *, ptr3dim

!  TEST26
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(2,2,1) .EQ. a2d_to_1d( 2 + (2-1)*2 + (1-1)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2,2,1)"
!    print *,ptr3dim(2,2,1)

!  TEST27
    tst_nbr = tst_nbr + 1
    ptr1dim(1:4) =>a2d(1:n, 1)
    result(tst_nbr) = all( ptr1dim .EQ. (/11,21,31,41/))
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:4) =>a2d(1:n, 1)"
!    print *, "a2d(1:n, 1):"
!    print *, a2d(1:n, 1)
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST28
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n-2, 1:n-2) =>a2d(1:n, 1)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a2d_to_1d, (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1:n-2, 1:n-2) =>a2d(1:n, 1)"
!    print *, "a2d(1:n, 1):"
!    print *, a2d(1:n, 1)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST29
    tst_nbr = tst_nbr + 1
    ptr3dim(1:n-2, 1:n-2, 2:4) =>a2d	
    result(tst_nbr) = all(ptr3dim .EQ. reshape(a2d_to_1d, (/2,2,3/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:n-2, 1:n-2, 2:4) =>a2d"
!    print *, "a2d:"
!    print *, a2d
!    print *, "ptr3dim:"
!    print *, ptr3dim

!  TEST30
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(1,1,2) .EQ. a2d_to_1d(1 + (1-1)*2 + (2-2)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1,1,2)"
!    print *,ptr3dim(1,1,2)

!  TEST31
    tst_nbr = tst_nbr + 1
    ptr1dim(1:n-2) =>a2d(2:4, 1)
    result(tst_nbr) = all(ptr1dim .EQ. a2d_to_1d(2:))
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:n-2, 1:n-2) =>a2d(2:4, 1)"
!    print *, "a2d(2:4, 1):"
!    print *, a2d(2:4, 1)
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST32
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n-2, 1:n-2) =>a2d(2:4, 1)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a2d_to_1d(2:), (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1:n-2, 1:n-2) =>a2d(2:4, 1)"
!    print *, "a2d(2:4, 1):"
!    print *, a2d(2:4, 1)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST33
    tst_nbr = tst_nbr + 1
    ptr3dim(1:n-2, 1:n-2, 1:3) =>a2d(2:4, 1)
    result(tst_nbr) = all(ptr3dim .EQ.  reshape(a2d_to_1d(2:), (/2,2,3/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:n-2, 1:n-2, 1:3) =>a2d(2:4, 1)"
!    print *, "a2d(2:4, 1):"
!    print *, a2d(2:4, 1)
!    print *, "ptr3dim:"
!    print *, ptr3dim

!  TEST34
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n-2, 1:n-2) =>a2d(:,2)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a2d_to_1d(5:),(/2,2/)))
!    print *,"TEST", tst_nbr
!    result(tst_nbr) = 
!    print *,"ptr2dim(1:n-2, 1:n-2) =>a2d(:,2)"
!    print *, "a2d(:,2):"
!    print *, a2d(:, 2)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST35
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr2dim(3, 2) .EQ. a2d_to_1d(3 + (2-1+2)*2)
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(3, 2)"
!    print *,ptr2dim(3, 2)

!  TEST36
    tst_nbr = tst_nbr + 1
    ptr1dim(3:n) => t2_inst%a2d(:,2)
    result(tst_nbr) = all(ptr1dim .EQ. a2d_to_1d(1 + (2-1)*4:))
!    print *,"TEST", tst_nbr
!    result(tst_nbr) = 
!    print *,"ptr1dim(3:n) => t2_inst%a2d(:,2)"
!    print *, "t2_inst%a2d(:,2):"
!    print *, t2_inst%a2d(:, 2)
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST37
    tst_nbr = tst_nbr + 1
    ptr2dim(2:n-1, 3:n) => t2_inst%a2d(:,2)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a2d_to_1d(1+(2-1)*4:), (/2,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(2:n-1, 3:n) => t2_inst%a2d(:,2)"
!    print *, "t2_inst%a2d(:,2:7):"
!    print *, t2_inst%a2d(:, 2:7)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST38
    tst_nbr = tst_nbr + 1
    ptr3dim(3:n, 2:n, 1:2) => t2_inst%a2d(:,2)
    result(tst_nbr) =  all(ptr3dim .EQ. reshape(a2d_to_1d(1 + (2-1)*4:), (/2,3,2/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(3:n, 2:n, 1:2) => t2_inst%a2d(:,2)"
!    print *, "t2_inst%a2d(:,2:13):"
!    print *, t2_inst%a2d(:, 2:13)
!    print *, "ptr3dim:"
!    print *, ptr3dim

!  TEST39
    tst_nbr = tst_nbr + 1
    t1_ptr2d(1:n-2, 1:n-2) =>t1_2d(2:4, 1)

     result(tst_nbr) = (t1_ptr2d(1,1)%i .EQ. t1_2d(2,1)%i) .AND. &
                       (t1_ptr2d(2,1)%i .EQ. t1_2d(3,1)%i) .AND. &
                       (t1_ptr2d(1,2)%i .EQ. t1_2d(4,1)%i) .AND. &
                       (t1_ptr2d(2,2)%i .EQ. t1_2d(1,2)%i)

!    print *,"TEST", tst_nbr
!    print *,"t1_ptr2d(1:n-2, 1:n-2) =>t1_2d(2:4, 1)"
!    print *, "t1_2d(2:4, 1):"
!    print *, t1_2d(2:4, 1)
!    print *, "t1_ptr2d:"
!    print *, t1_ptr2d`			! this print produces incorrect results

 !===============================================

!  TEST40
    tst_nbr = tst_nbr + 1
    ptr1dim(1:n) =>a3d(:,:,1)
    result(tst_nbr) = all(ptr1dim .EQ. a3d_to_1d)
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim(1:n) =>a3d(:,:,1)"
!    print *, "a3d(:,:,1):"
!    print *, a3d(:,:,1)
!    print *, "ptr1dim:"
!    print *, ptr1dim

!  TEST41
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr1dim(3) .EQ. a3d(3,1,1)
!    print *,"TEST", tst_nbr
!    print *,"ptr1dim4"
!    print *,ptr1dim(3)

!  TEST42
    tst_nbr = tst_nbr + 1
    ptr2dim(1:n, 1:n) =>a3d(:,:,1)
    result(tst_nbr) = all(ptr2dim .EQ. reshape(a3d_to_1d, (/4,4/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1:n, 1:n) =>a3d(:,:,1)"
!    print *, "a3d(:,:,1):"
!    print *, a3d(:,:,1)
!    print *, "ptr2dim:"
!    print *, ptr2dim

!  TEST43
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr2dim(1,4) .EQ. a3d_to_1d(1 + (4-1)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr2dim(1,4)"
!    print *,ptr2dim(1,4)

!  TEST44
    tst_nbr = tst_nbr + 1
    ptr3dim(1:2, 1:2, 1:3) =>a3d(:,:,1)
    result(tst_nbr) = all(ptr3dim .EQ. reshape(a3d_to_1d, (/2,2,3/)))
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(1:2, 1:2, 1:3) =>a3d(:,:,1)"
!    print *, "a3d(:,:,1):"
!    print *, a3d(:,:,1)
!    print *, "ptr3dim:"
!    print *, ptr3dim
! print *, result(tst_nbr)  		! causes crash

!  TEST45
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = ptr3dim(2,2,3) .EQ. a3d_to_1d(2 + (2-1)*2 + (3-1)*4)
!    print *,"TEST", tst_nbr
!    print *,"ptr3dim(2,2,3)"
!    print *,ptr3dim(2,2,3)

!  TEST46
    tst_nbr = tst_nbr + 1
   cplxptr2d(1:n, 1:n-2) => cplx3d(:,2:3,1)
   result(tst_nbr) = all(cplxptr2d .EQ.  reshape(cplx3d_to_1d(1 + (2-1)*4:), (/4,2/)))
!    print *,"TEST", tst_nbr
!    print *,"cplxptr2d(1:n, 1:n) =>cplx3d(:,:,1)"
!    print *, "cplx3d(:,:,1):"
!    print *, cplx3d(:,:,1)
!    print *, "cplxptr2d:"
!    print *, cplxptr2d

!  TEST47
    tst_nbr = tst_nbr + 1
    result(tst_nbr) = cplxptr2d(2,2) .EQ. cplx3d_to_1d(2 + 2*4)
!    print *,"TEST", tst_nbr
!    print *,"cplxptr2d(2,2)"
!    print *,cplxptr2d(2,2)

  call check(result, expect, NBR_TSTS)
end program
