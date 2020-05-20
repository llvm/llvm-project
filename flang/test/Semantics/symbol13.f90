! RUN: %S/test_symbols.sh %s %t %f18
! Old-style "*length" specifiers (R723)

!DEF: /f1 (Function) Subprogram CHARACTER(1_8,1)
!DEF: /f1/x1 INTENT(IN) ObjectEntity CHARACTER(2_4,1)
!DEF: /f1/x2 INTENT(IN) ObjectEntity CHARACTER(3_4,1)
character*1 function f1(x1, x2)
 !DEF: /f1/n PARAMETER ObjectEntity INTEGER(4)
 integer, parameter :: n = 2
 !REF: /f1/n
 !REF: /f1/x1
 !REF: /f1/x2
 !DEF: /f1/len INTRINSIC (Function) ProcEntity
 character*(n), intent(in) :: x1, x2*(len(x1)+1)
 !DEF: /f1/t DerivedType
 type :: t
  !REF: /f1/len
  !REF: /f1/x2
  !DEF: /f1/t/c1 ObjectEntity CHARACTER(4_4,1)
  !DEF: /f1/t/c2 ObjectEntity CHARACTER(6_8,1)
  character*(len(x2)+1) :: c1, c2*6
 end type t
end function f1
