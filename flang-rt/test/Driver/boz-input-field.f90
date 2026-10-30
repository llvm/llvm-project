! B/O/Z input should consume its entire fixed-width field, including trailing
! blanks and an early terminating separator, so that the next edit
! descriptor starts after the field.  READ(SIZE=) should count each character
! of the field once.
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: int4: 1F "XY"
! CHECK: int8: 1F "XY"
! CHECK: int16: 1F "XY"
! CHECK: char: 4142 "XY"
! CHECK: comma: 1F 2
! CHECK: octal: 17 "XY"
! CHECK: binary: 5 "XY"
! CHECK: size: 1F 6
! CHECK: PASS

program boz_input_field
  implicit none
  integer :: i4, iu, n
  integer(8) :: i8
  integer(16) :: i16
  character(4) :: c4
  character(2) :: c2
  character(:), allocatable :: s

  s = '1F    XY'
  read (s, '(z6,a2)') i4, c2
  print '(a,z0,1x,3a)', 'int4: ', i4, '"', c2, '"'
  read (s, '(z6,a2)') i8, c2
  print '(a,z0,1x,3a)', 'int8: ', i8, '"', c2, '"'
  read (s, '(z6,a2)') i16, c2
  print '(a,z0,1x,3a)', 'int16: ', i16, '"', c2, '"'

  s = '4142    XY'
  read (s, '(z8,a2)') c4, c2
  print '(a,z0,1x,3a)', 'char: ', transfer(c4, 0), '"', c2, '"'

  s = '1f,2'
  read (s, '(z4,i2)') i16, i4
  print '(a,z0,1x,i0)', 'comma: ', i16, i4

  s = '17  XY'
  read (s, '(o4,a2)') i16, c2
  print '(a,o0,1x,3a)', 'octal: ', i16, '"', c2, '"'

  s = '101  XY'
  read (s, '(b5,a2)') i16, c2
  print '(a,i0,1x,3a)', 'binary: ', i16, '"', c2, '"'

  open (newunit=iu, status='scratch', form='formatted')
  write (iu, '(a)') '  1F  XY'
  rewind (iu)
  read (iu, '(z6)', advance='no', size=n) i16
  print '(a,z0,1x,i0)', 'size: ', i16, n
  close (iu)

  print '(a)', 'PASS'
end program
