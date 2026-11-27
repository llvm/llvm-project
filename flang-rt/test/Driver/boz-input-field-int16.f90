! B/O/Z input into an INTEGER(16) item should consume its entire fixed-width
! field, including trailing blanks, so that the next edit descriptor starts
! after the field.
!
! https://github.com/llvm/llvm-project/issues/243701
! UNSUPPORTED: system-windows
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s

! CHECK: int16: 1F "XY"
! CHECK: PASS

program boz_input_field_int16
  implicit none
  integer(16) :: i16
  character(2) :: c2
  character(:), allocatable :: s

  s = '1F    XY'
  read (s, '(z6,a2)') i16, c2
  print '(a,z0,1x,3a)', 'int16: ', i16, '"', c2, '"'

  print '(a)', 'PASS'
end program
