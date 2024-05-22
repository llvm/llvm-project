! RUN: %python %S/test_symbols.py %s %flang_fc1
! References to generic functions with forward-referenced specifics.
!DEF: /m Module
module m
contains
 !DEF: /m/specific4 PUBLIC (Function) Subprogram INTEGER(4)
 !DEF: /m/specific4/x INTENT(IN) ObjectEntity INTEGER(4)
 integer function specific4(x)
  !REF: /m/specific4/x
  integer, intent(in) :: x(*)
 end function
 !DEF: /m/test PUBLIC (Subroutine) Subprogram
 !DEF: /m/test/specific1 EXTERNAL (Function) Subprogram INTEGER(4)
 subroutine test (specific1)
  !DEF: /m/test/generic (Function) Generic
  interface generic
   !REF: /m/test/specific1
   procedure :: specific1
   !DEF: /m/test/specific2 EXTERNAL, PURE (Function) Subprogram INTEGER(4)
   procedure :: specific2
   !DEF: /m/test/specific3 EXTERNAL (Function) Subprogram INTEGER(4)
   procedure :: specific3
   !DEF: /m/test/specific4 EXTERNAL (Function) Subprogram INTEGER(4)
   procedure :: specific4
  end interface
  interface
   !REF: /m/test/specific1
   !DEF: /m/test/specific1/x INTENT(IN) ObjectEntity INTEGER(4)
   integer function specific1(x)
    !REF: /m/test/specific1/x
    integer, intent(in) :: x
   end function
   !REF: /m/test/specific2
   !DEF: /m/test/specific2/x INTENT(IN) ObjectEntity INTEGER(4)
   !DEF: /m/test/specific2/y INTENT(IN) ObjectEntity INTEGER(4)
   pure integer function specific2(x, y)
    !REF: /m/test/specific2/x
    !REF: /m/test/specific2/y
    integer, intent(in) :: x, y
   end function
   !REF: /m/test/specific3
   !DEF: /m/test/specific3/x INTENT(IN) ObjectEntity INTEGER(4)
   !DEF: /m/test/specific3/y INTENT(IN) ObjectEntity INTEGER(4)
   integer function specific3(x, y)
    !REF: /m/test/generic
    import :: generic
    !REF: /m/test/specific3/x
    !REF: /m/test/specific3/y
    !REF: /m/test/specific2
    integer, intent(in) :: x, y(generic(1, x))
   end function
   !REF: /m/test/specific4
   !DEF: /m/test/specific4/x INTENT(IN) ObjectEntity INTEGER(4)
   integer function specific4(x)
    !REF: /m/test/specific4/x
    integer, intent(in) :: x(:)
   end function
  end interface
  !REF: /m/test/specific4
  print *, generic([1])
 end subroutine
end module
