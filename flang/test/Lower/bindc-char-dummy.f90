! Test that a BIND(C) procedure does not pass a hidden length for a character
! dummy argument. A BIND(C) procedure uses the C calling convention, which has
! no such argument, and semantics already requires a BIND(C) character dummy to
! have a length of one, so the length carries no information.

! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine call_bindc_char_dummy(c, a)
  interface
    subroutine takes_char(s) bind(C, name="takes_char")
      character(kind=1, len=1), intent(in) :: s
    end subroutine
    subroutine takes_char_array(s) bind(C, name="takes_char_array")
      character(kind=1, len=1), intent(in) :: s(*)
    end subroutine
  end interface
  character(kind=1, len=1) :: c
  character(kind=1, len=1) :: a(10)
  call takes_char(c)
  call takes_char_array(a)
end subroutine
! CHECK-LABEL:   func.func @_QPcall_bindc_char_dummy(
! CHECK:           fir.call @takes_char(%{{[^)]*}}) proc_attrs<bind_c>{{.*}} : (!fir.ref<!fir.char<1>>) -> ()
! CHECK:           fir.call @takes_char_array(%{{[^)]*}}) proc_attrs<bind_c>{{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>) -> ()

subroutine defines_char(s) bind(C, name="defines_char")
  character(kind=1, len=1), intent(inout) :: s
  s = 'x'
end subroutine
! CHECK-LABEL:   func.func @defines_char(
! CHECK-SAME:      %{{[^:]*}}: !fir.ref<!fir.char<1>>
