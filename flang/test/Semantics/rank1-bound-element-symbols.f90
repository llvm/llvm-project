! This test is eventually meant to test several contexts where a 
! rank1BoundElement node is used. That is currently limited to
! ExplicitShapeBoundsSpec, but will later include every other context
! where one can use rank-1 integer array bounds instead of past syntax.
! This includes assumed shape bounds, pointer assignment with bounds remapping,
! and allocate statements.

! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s --check-prefix=SYMBOLS
! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s --check-prefix=UNPARSE

subroutine s(n)
  integer, intent(in) :: n(3)
  !SYMBOLS: a {{.*}}: ObjectEntity type: REAL(4) shape: 1_8:rank1BoundElement(__builtin_int(n,kind=8),dim=1),1_8:rank1BoundElement(__builtin_int(n,kind=8),dim=2),1_8:rank1BoundElement(__builtin_int(n,kind=8),dim=3)
  real :: a(n)
  a = 0.0
end subroutine

subroutine s2
  ! A zero-size bounds array overrides the DIMENSION attribute and declares a
  ! scalar (size=4, no shape), rather than a size=5*4 array.
  !SYMBOLS: z size=4 {{.*}}: ObjectEntity type: INTEGER(4)
  integer, dimension(5) :: z(1 : [integer ::])
end

! -fdebug-unparse-with-symbols intentionally reproduces the original bound syntax 
! rather than the synthesized rank1BoundElement node; this confirms the construct 
! still round-trips through that action with its symbol annotations.
!UNPARSE: subroutine s (n)
!UNPARSE: integer, intent(in) :: n(3)
!UNPARSE: !DEF: /s/a ObjectEntity REAL(4)
!UNPARSE: real a(n)
