! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! A non-ASCII CHARACTER conversion between kinds must not be folded (in
! particular not to blanks); it is left to the run-time conversion.

! CHECK-LABEL: func.func @_QPnonascii(
subroutine nonascii(first)
  character(kind=4, len=8) :: first
  ! CHECK: %[[LIT:.*]] = fir.address_of(@[[NONASCII:_QQcl[^)]*]]) : !fir.ref<!fir.char<1,9>>
  ! CHECK: %[[LITDECL:.*]]:2 = hlfir.declare %[[LIT]]
  ! CHECK: fir.char_convert %[[LITDECL]]#0 for %{{.*}} to %{{.*}} : !fir.ref<!fir.char<1,9>>, index, !fir.ref<!fir.char<4,?>>
  first = '3.14 = π'
end subroutine

! CHECK-LABEL: func.func @_QPascii(
subroutine ascii(first)
  character(kind=4, len=8) :: first
  ! CHECK-NOT: fir.char_convert
  ! CHECK: fir.address_of(@{{.*}}) : !fir.ref<!fir.char<4,8>>
  first = '3.14 = p'
end subroutine

! CHECK: fir.global linkonce @[[NONASCII]] constant : !fir.char<1,9> {
! CHECK:   fir.string_lit "3.14 = \CF\80"(9) : !fir.char<1,9>
