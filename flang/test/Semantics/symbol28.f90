! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /m1 Module
module m1
 !DEF: /m1/s PUBLIC (Subroutine) Generic
 interface s
  !DEF: /m1/s MODULE (Subroutine) Subprogram
  module subroutine s
  end subroutine
  !DEF: /m1/s2 MODULE, PUBLIC (Subroutine) Subprogram
  !DEF: /m1/s2/j INTENT(IN) ObjectEntity INTEGER(4)
  module subroutine s2 (j)
   !REF: /m1/s2/j
   integer, intent(in) :: j
  end subroutine
 end interface
contains
 !DEF: /m1/s MODULE (Subroutine) Subprogram
 module subroutine s
 end subroutine
 !REF: /m1/s2
 module procedure s2
 end procedure
 !DEF: /m1/test PUBLIC (Subroutine) Subprogram
 subroutine test
  !REF: /m1/s
  call s
  !REF: /m1/s2
  call s(1)
 end subroutine
end module
!DEF: /m2 Module
module m2
 !DEF: /m2/s PUBLIC (Subroutine) Generic
 interface s
  !DEF: /m2/s MODULE (Subroutine) Subprogram
  module subroutine s
  end subroutine
  !DEF: /m2/s2 MODULE, PUBLIC (Subroutine) Subprogram
  !DEF: /m2/s2/j INTENT(IN) ObjectEntity INTEGER(4)
  module subroutine s2 (j)
   !REF: /m2/s2/j
   integer, intent(in) :: j
  end subroutine
 end interface
contains
 !REF:/m2/s
 module procedure s
 end procedure
 !DEF: /m2/s2 MODULE, PUBLIC (Subroutine) Subprogram
 !DEF: /m2/s2/j INTENT(IN) ObjectEntity INTEGER(4)
 module subroutine s2 (j)
  !REF: /m2/s2/j
  integer, intent(in) :: j
 end subroutine
 !DEF: /m2/test PUBLIC (Subroutine) Subprogram
 subroutine test
  !REF: /m2/s
  call s
  !REF: /m2/s2
  call s(1)
 end subroutine
end module
