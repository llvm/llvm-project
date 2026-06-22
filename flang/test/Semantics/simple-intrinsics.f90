! RUN: %python %S/test_symbols.py %s %flang_fc1
! Fortran 2023 classifies MVBITS, SPLIT, and TOKENIZE as SIMPLE
! intrinsic subroutines. Verify that their symbols resolve with the
! SIMPLE attribute. MVBITS is also ELEMENTAL.

!DEF: /expect_simple_intrinsics (Subroutine) Subprogram
!DEF: /expect_simple_intrinsics/string INTENT(IN) ObjectEntity CHARACTER(*,1)
!DEF: /expect_simple_intrinsics/set INTENT(IN) ObjectEntity CHARACTER(*,1)
!DEF: /expect_simple_intrinsics/from INTENT(IN) ObjectEntity INTEGER(4)
!DEF: /expect_simple_intrinsics/to INTENT(INOUT) ObjectEntity INTEGER(4)
subroutine expect_simple_intrinsics (string, set, from, to)
 !REF: /expect_simple_intrinsics/string
 !REF: /expect_simple_intrinsics/set
 character(len=*), intent(in) :: string, set
 !REF: /expect_simple_intrinsics/from
 integer, intent(in) :: from
 !REF: /expect_simple_intrinsics/to
 integer, intent(inout) :: to
 !DEF: /expect_simple_intrinsics/pos ObjectEntity INTEGER(4)
 integer pos
 !DEF: /expect_simple_intrinsics/tokens ALLOCATABLE ObjectEntity CHARACTER(:,1)
 character(len=:), allocatable :: tokens(:)
 !DEF: /expect_simple_intrinsics/first ALLOCATABLE ObjectEntity INTEGER(4)
 !DEF: /expect_simple_intrinsics/last ALLOCATABLE ObjectEntity INTEGER(4)
 integer, allocatable :: first(:), last(:)
 !REF: /expect_simple_intrinsics/pos
 pos = 1
 !DEF: /expect_simple_intrinsics/split INTRINSIC, SIMPLE (Subroutine) ProcEntity
 !REF: /expect_simple_intrinsics/string
 !REF: /expect_simple_intrinsics/set
 !REF: /expect_simple_intrinsics/pos
 call split(string, set, pos)
 !DEF: /expect_simple_intrinsics/tokenize INTRINSIC, SIMPLE (Subroutine) ProcEntity
 !REF: /expect_simple_intrinsics/string
 !REF: /expect_simple_intrinsics/set
 !REF: /expect_simple_intrinsics/tokens
 call tokenize(string, set, tokens)
 !REF: /expect_simple_intrinsics/tokenize
 !REF: /expect_simple_intrinsics/string
 !REF: /expect_simple_intrinsics/set
 !REF: /expect_simple_intrinsics/first
 !REF: /expect_simple_intrinsics/last
 call tokenize(string, set, first, last)
 !DEF: /expect_simple_intrinsics/mvbits ELEMENTAL, INTRINSIC, SIMPLE (Subroutine) ProcEntity
 !REF: /expect_simple_intrinsics/from
 !REF: /expect_simple_intrinsics/to
 call mvbits(from, 0, 1, to, 2)
end subroutine

