! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY-NOT: mif.stop

subroutine stop_test()
  ! COARRAY: mif.stop : ()
  stop
end subroutine

subroutine stop_code1()
  integer int_code
  ! COARRAY: mif.stop code %[[CODE:.*]] : (i32)
  stop int_code
end subroutine

subroutine stop_code2()
  ! COARRAY: mif.stop code %[[CODE:.*]] : (i32)
  stop ((5 + 8) * 2)
end subroutine

subroutine stop_code_char1()
  character(len=128) char_code
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C128:.*]] : (!fir.ref<!fir.char<1,128>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.stop code %[[CODE]] : (!fir.boxchar<1>)
  stop char_code
end subroutine

subroutine stop_code_char2()
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C1:.*]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.stop code %[[CODE]] : (!fir.boxchar<1>)
  stop 'c'
end subroutine

subroutine stop_code_char3()
  ! COARRAY: %[[CODE:.*]] = fir.emboxchar %[[VAL:.*]]#0, %[[C14:.*]] : (!fir.ref<!fir.char<1,14>>, index) -> !fir.boxchar<1>
  ! COARRAY: mif.stop code %[[CODE]] : (!fir.boxchar<1>)
  stop ('program failed')
end subroutine

subroutine stop_code_quiet1()
  integer int_code
  logical bool
  ! COARRAY mif.stop
  stop int_code, quiet=bool
end subroutine

subroutine stop_code_quiet2()
  integer int_code
  ! COARRAY mif.stop code %[[CODE:.*]] quiet %true : (i32, i1)
  stop int_code, quiet=.true.
end subroutine

subroutine stop_code_quiet3()
  integer int_code
  ! COARRAY mif.stop code %[[CODE:.*]] quiet %false : (i32, i1)
  stop (int_code), quiet=.false.
end subroutine
