! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell
! Test host association in module subroutine and internal subroutine.

!DEF: /m Module
module m
 !DEF: /m/t PUBLIC DerivedType
 type :: t
 end type
 !REF: /m/t
 !DEF: /m/x PUBLIC ObjectEntity TYPE(t)
 type(t) :: x
 interface
  !DEF: /m/s3 MODULE, PUBLIC (Subroutine) Subprogram
  !DEF: /m/s3/y ObjectEntity TYPE(t)
  module subroutine s3(y)
   !REF: /m/t
   !REF: /m/s3/y
   type(t) :: y
  end subroutine
 end interface
contains
 !DEF: /m/s PUBLIC (Subroutine) Subprogram
 subroutine s
  !REF: /m/t
  !DEF: /m/s/y ObjectEntity TYPE(t)
  type(t) :: y
  !REF: /m/s/y
  !REF: /m/x
  y = x
  !DEF: /m/s/s (Subroutine) HostAssoc
  call s
 contains
  !DEF: /m/s/s2 (Subroutine) Subprogram
  subroutine s2
   !REF: /m/x
   !REF: /m/s/y
   !REF: /m/t
   !REF: /m/s/s
   import, only: x, y, t, s
   !REF: /m/t
   !DEF: /m/s/s2/z ObjectEntity TYPE(t)
   type(t) :: z
   !REF: /m/s/s2/z
   !REF: /m/x
   z = x
   !REF: /m/s/s2/z
   !DEF: /m/s/s2/y HostAssoc TYPE(t)
   z = y
   !REF: /m/s/s
   call s
  end subroutine
 end subroutine
end module
