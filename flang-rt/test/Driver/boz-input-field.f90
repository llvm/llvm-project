! B/O/Z input should consume its entire fixed-width field, including trailing
! blanks and an early terminating separator, so that the next edit
! descriptor starts after the field.  READ(SIZE=) should count each character
! of the field once.
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: int4: 1F "XY"
! CHECK: int8: 1F "XY"
! CHECK: char: 4142 "XY"
! CHECK: comma: 1F 2
! CHECK: octal: 17 "XY"
! CHECK: binary: 5 "XY"
! CHECK: 2fields: 1F 2E
! CHECK: size: 1F 6
! CHECK: second: 2A "ZW" 8
! CHECK: size3: 1 2 3 12 "XY"
! CHECK: bzcomma: 1F 3
! CHECK: PASS

program boz_input_field
  implicit none
  integer :: i4, j4, k4, iu, n
  integer(8) :: i8
  character(4) :: c4
  character(2) :: c2
  character(:), allocatable :: s

  s = '1F    XY'
  read (s, '(z6,a2)') i4, c2
  print '(a,z0,1x,3a)', 'int4: ', i4, '"', c2, '"'
  read (s, '(z6,a2)') i8, c2
  print '(a,z0,1x,3a)', 'int8: ', i8, '"', c2, '"'

  ! B/O/Z editing of a CHARACTER item is an extension.
  s = '4142    XY'
  read (s, '(z8,a2)') c4, c2
  print '(a,z0,1x,3a)', 'char: ', transfer(c4, 0), '"', c2, '"'

  ! A separator ending a fixed-width field early is an extension.
  s = '1f,2'
  read (s, '(z4,i2)') i4, j4
  print '(a,z0,1x,i0)', 'comma: ', i4, j4

  s = '17  XY'
  read (s, '(o4,a2)') i4, c2
  print '(a,o0,1x,3a)', 'octal: ', i4, '"', c2, '"'

  s = '101  XY'
  read (s, '(b5,a2)') i4, c2
  print '(a,i0,1x,3a)', 'binary: ', i4, '"', c2, '"'

  ! A second B/O/Z field after a short value.
  s = '1F    2E'
  read (s, '(z6,z2)') i4, j4
  print '(a,z0,1x,z0)', '2fields: ', i4, j4

  open (newunit=iu, status='scratch', form='formatted')
  write (iu, '(a)') '  1F  XY'
  rewind (iu)
  read (iu, '(z6)', advance='no', size=n) i4
  print '(a,z0,1x,i0)', 'size: ', i4, n
  close (iu)

  ! A second non-advancing READ continues the record, so the field does not
  ! begin at the left tab limit of zero.
  open (newunit=iu, status='scratch', form='formatted')
  write (iu, '(a)') '  1F  XY 2A   ZW'
  rewind (iu)
  read (iu, '(z6,a2)', advance='no') i4, c2
  read (iu, '(z6,a2)', advance='no', size=n) i4, c2
  print '(a,z0,1x,3a,1x,i0)', 'second: ', i4, '"', c2, '"', n
  close (iu)

  ! SIZE= accumulated over several fields, and with BZ and a separator.
  open (newunit=iu, status='scratch', form='formatted')
  write (iu, '(a)') '1   2   3   XY'
  write (iu, '(a)') '1F,  XY'
  rewind (iu)
  read (iu, '(3z4)', advance='no', size=n) i4, j4, k4
  read (iu, '(a2)') c2
  print '(a,3(z0,1x),i0,1x,3a)', 'size3: ', i4, j4, k4, n, '"', c2, '"'
  read (iu, '(bz,z6)', advance='no', size=n) i4
  read (iu, '(a)')
  print '(a,z0,1x,i0)', 'bzcomma: ', i4, n
  close (iu)

  print '(a)', 'PASS'
end program
