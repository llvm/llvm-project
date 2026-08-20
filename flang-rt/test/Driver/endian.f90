! UNSUPPORTED: offload-cuda

! Verify endian conversion for unformatted stream I/O with
! CONVERT='BIG_ENDIAN'. The test writes values in big-endian
! format, reads raw bytes back, applies SwapEndianness(), and
! checks that the original values are restored.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: PASS
program main
  implicit none
  call test2()
  call test4()
  call test8()
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

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='big_endian')
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

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='big_endian')
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

  open(10,file='test.dat',form='unformatted',access='stream',status='unknown', convert='big_endian')
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
