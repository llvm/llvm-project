! Formatted I/O of enumeration type items: input values out of range and
! edit descriptors other than I, B, O and Z are I/O errors (F2023 12.6.3 p6,
! 13.7.2.2 p6).

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

program enumeration_io
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type :: holder
    integer :: n
    type(color) :: c
  end type
  type(color) :: c, a(3), empty(0)
  type(holder) :: h
  character(8) :: buf
  character(100) :: msg
  integer :: ios

  ! CHECK: in range: 0 3
  buf = '3'
  read(buf, '(I1)', iostat=ios) c
  print '(a,i0,1x,i0)', 'in range: ', ios, int(c)

  ! CHECK: zero: T 3
  buf = '0'
  read(buf, '(I1)', iostat=ios) c
  print '(a,l1,1x,i0)', 'zero: ', ios /= 0, int(c)

  ! CHECK: too big: T 3 Input value 4 is out of range for enumeration type 'color' (must be from 1 to 3)
  buf = '4'
  read(buf, '(I1)', iostat=ios, iomsg=msg) c
  print '(a,l1,1x,i0,1x,a)', 'too big: ', ios /= 0, int(c), trim(msg)

  ! CHECK: binary: T 3
  buf = '101'
  read(buf, '(B3)', iostat=ios) c
  print '(a,l1,1x,i0)', 'binary: ', ios /= 0, int(c)

  ! CHECK: array: T 3 2 3
  a = [red, green, blue]
  buf = '3 9 1'
  read(buf, '(3I2)', iostat=ios) a
  print '(a,l1,3(1x,i0))', 'array: ', ios /= 0, int(a)

  ! CHECK: component: T 7 2
  h = holder(1, green)
  buf = ' 7 5'
  read(buf, '(2I2)', iostat=ios) h
  print '(a,l1,1x,i0,1x,i0)', 'component: ', ios /= 0, h%n, int(h%c)

  ! CHECK: G input: T 2
  buf = '1'
  read(buf, '(G1.0)', iostat=ios) h%c
  print '(a,l1,1x,i0)', 'G input: ', ios /= 0, int(h%c)

  ! CHECK: G output: T
  write(buf, '(G0)', iostat=ios) c
  print '(a,l1)', 'G output: ', ios /= 0

  ! CHECK: I output: 3
  write(buf, '(I0)') c
  print '(a,a)', 'I output: ', trim(buf)

  ! A zero-size array must not end the transfer early.
  ! CHECK: after empty: 0 1
  buf = '1'
  read(buf, '(I1)', iostat=ios) empty, c
  print '(a,i0,1x,i0)', 'after empty: ', ios, int(c)
end program
