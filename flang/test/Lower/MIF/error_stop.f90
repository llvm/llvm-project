! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY-NOT: mif.error_stop

subroutine error_stop_test()
  ! COARRAY: mif.error_stop : ()
  error stop
end subroutine

subroutine error_stop_code1()
  integer int_code
  ! COARRAY: mif.error_stop code %[[CODE:.*]] : (i32)
  error stop int_code
end subroutine

subroutine error_stop_code2()
  ! COARRAY: mif.error_stop code %[[CODE:.*]] : (i32)
  error stop ((5 + 8) * 2)
end subroutine

subroutine error_stop_code_char1()
  character(len=128) char_code
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C128:.*]] : (!fir.ref<!fir.char<1,128>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.error_stop code %[[CODE]] : (!fir.boxchar<1>)
  error stop char_code
end subroutine

subroutine error_stop_code_char2()
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C1:.*]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.error_stop code %[[CODE]] : (!fir.boxchar<1>)
  error stop 'c'
end subroutine

subroutine error_stop_code_char3()
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C14:.*]] : (!fir.ref<!fir.char<1,14>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.error_stop code %[[CODE]] : (!fir.boxchar<1>)
  error stop ('program failed')
end subroutine

subroutine error_stop_code_quiet1()
  integer int_code
  logical bool
  ! COARRAY mif.error_stop
  error stop int_code, quiet=bool
end subroutine

subroutine error_stop_code_quiet2()
  integer int_code
  ! COARRAY mif.error_stop code %[[CODE:.*]] quiet %true : (i32, i1)
  error stop int_code, quiet=.true.
end subroutine

subroutine error_stop_code_quiet3()
  integer int_code
  ! COARRAY mif.error_stop code %[[CODE:.*]] quiet %false : (i32, i1)
  error stop (int_code), quiet=.false.
end subroutine
