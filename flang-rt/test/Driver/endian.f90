! UNSUPPORTED: offload-cuda

! Verify endian conversion for unformatted stream I/O with
! CONVERT='SWAP'. The test writes values with byte swapping,
! reads raw bytes back, applies SwapEndianness(), and checks
! that the original values are restored.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: PASS
program main
  implicit none
  call test2()
  call test4()
  call test8()
  call testEmptyItem()
  call testReadSwapped()
  print *,'PASS'

end program main
subroutine SwapEndianness(data, bytes, elementBytes)
  implicit none
  integer(1) data(0:*), tmp
  integer(8) bytes, elementBytes, half
  integer(8) j,k
  half = elementBytes/2
  do j = 0, bytes - elementBytes, elementBytes
     do k = 0, half-1
        tmp = data(j + k)
        data(j + k) = data(j + elementBytes - 1 - k)
        data(j + elementBytes - 1 - k) = tmp
     end do
  end do
end subroutine
subroutine test2()
  implicit none
  integer(2) i2(2)
  integer(1) i8(4)
  i2 = (/1,2/)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='swap')
  write(10) i2
  close(10)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown')
  read(10) i8
  close(10)

  call SwapEndianness(i8, 4_8, 2_8)

  if (.not. all (i2 == transfer(i8, i2))) then
     stop
  endif
end subroutine
subroutine test4()
  implicit none
  real(4) f32(2)
  integer(1) i8(8)
  f32 = (/1,2/)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='swap')
  write(10) f32
  close(10)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown')
  read(10) i8
  close(10)

  call SwapEndianness(i8, 8_8, 4_8)

  if (.not. all (f32 == transfer(i8, f32))) then
     stop
  endif
end subroutine
subroutine test8()
  implicit none
  real(8) f64(2)
  integer(1) i8(16)
  f64 = (/1,2/)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='swap')
  write(10) f64
  close(10)

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown')
  read(10) i8
  close(10)

  call SwapEndianness(i8, 16_8, 8_8)

  if (.not. all (f64 == transfer(i8, f64))) then
     stop
  endif
end subroutine
! A zero-size array followed by another item, with IOSTAT=.
! Fails without the change: the scalar is neither written nor read.
subroutine testEmptyItem()
  implicit none
  integer(4), allocatable :: empty(:)
  integer(4) :: x, y, ios, n
  allocate(empty(0))
  x = 123456789
  open(10,file='test.dat',form='unformatted',access='stream',status='replace',convert='swap')
  write(10,iostat=ios) empty, x
  inquire(10,size=n)
  if (ios /= 0 .or. n /= 4) error stop 'empty item: write'
  rewind(10)
  y = -1
  read(10,iostat=ios) empty, y
  close(10)
  if (ios /= 0 .or. y /= x) error stop 'empty item: read'
end subroutine
! Swapped READ of arrays whose swap unit is smaller than the element
! (CHARACTER(KIND=4), COMPLEX), plus a strided target.
subroutine testReadSwapped()
  implicit none
  character(kind=4,len=3) :: c(3), c2(3)
  complex(4) :: z(3), z2(3)
  integer(4) :: i(6), i2(6)
  c = [4_'abc', 4_'def', 4_'ghi']
  z = [(1.0,2.0), (3.0,4.0), (5.0,6.0)]
  i = [1, 2, 3, 4, 5, 6]
  open(10,file='test.dat',form='unformatted',access='stream',status='replace',convert='swap')
  write(10) c, z, i(1:6:2)
  rewind(10)
  i2 = -1
  read(10) c2, z2, i2(1:6:2)
  close(10)
  if (any(c2 /= c) .or. any(z2 /= z)) error stop 'swapped read'
  if (any(i2(1:6:2) /= i(1:6:2)) .or. any(i2(2:6:2) /= -1)) error stop 'strided read'
end subroutine
