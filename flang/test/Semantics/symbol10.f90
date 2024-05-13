! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /m1 Module
module m1
contains
 !DEF: /m1/foo_complex PUBLIC (Subroutine) Subprogram
 !DEF: /m1/foo_complex/z ObjectEntity COMPLEX(4)
 subroutine foo_complex (z)
  !REF: /m1/foo_complex/z
  complex z
 end subroutine
end module
!DEF: /m2 Module
module m2
 !REF: /m1
 use :: m1
 !DEF: /m2/foo PUBLIC (Subroutine) Generic
 interface foo
  !DEF: /m2/foo_int PUBLIC (Subroutine) Subprogram
  module procedure :: foo_int
  !DEF: /m2/foo_real EXTERNAL, PUBLIC (Subroutine) Subprogram
  procedure :: foo_real
  !DEF: /m2/foo_complex PUBLIC (Subroutine) Use
  procedure :: foo_complex
 end interface
 interface
  !REF: /m2/foo_real
  !DEF: /m2/foo_real/r ObjectEntity REAL(4)
  subroutine foo_real (r)
   !REF: /m2/foo_real/r
   real r
  end subroutine
 end interface
contains
 !REF: /m2/foo_int
 !DEF: /m2/foo_int/i ObjectEntity INTEGER(4)
 subroutine foo_int (i)
  !REF: /m2/foo_int/i
  integer i
 end subroutine
end module
