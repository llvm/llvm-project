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
  !SYMBOLS: a {{.*}}: ObjectEntity type: REAL(4) shape: 1_8:%ROBE(__builtin_int(n,kind=8),dim=1),1_8:%ROBE(__builtin_int(n,kind=8),dim=2),1_8:%ROBE(__builtin_int(n,kind=8),dim=3)
  real :: a(n)
  a = 0.0
end subroutine

! A constant bounds array is folded into ordinary per-dimension constant
! bounds when the entity is declared, so consumers that read bounds directly
! (here EQUIVALENCE storage-sequence offsets) see the same shape as the
! equivalent explicit-shape-spec-list: a(2,3) is element 8, so b starts at
! byte 28, and a(1,2) of a(0:2,0:3) is element 8 as well.
subroutine s_equivalence
  !SYMBOLS: a size=48 offset=0: ObjectEntity type: REAL(4) shape: 1_8:3_8,1_8:4_8
  !SYMBOLS: b size=48 offset=28: ObjectEntity type: REAL(4) shape: 1_8:12_8
  real :: a([3,4]), b(12)
  equivalence (a(2,3), b(1))
end subroutine
subroutine s_equivalence_lb
  !SYMBOLS: a size=48 offset=0: ObjectEntity type: REAL(4) shape: 0_8:2_8,0_8:3_8
  !SYMBOLS: b size=48 offset=28: ObjectEntity type: REAL(4) shape: 1_8:12_8
  real :: a([0,0]:[2,3]), b(12)
  equivalence (a(1,2), b(1))
end subroutine

subroutine s2
  ! A zero-size bounds array overrides the DIMENSION attribute and declares a
  ! scalar (size=4, no shape), rather than a size=5*4 array.
  !SYMBOLS: z size=4 {{.*}}: ObjectEntity type: INTEGER(4)
  integer, dimension(5) :: z(1 : [integer ::])
end

subroutine s3(x)
  type t
    integer :: c
  end type
  type(t), intent(in) :: x(2)
  ! When a scalar component follows the array parent, the extracted element
  ! subscript must land on the parent (x(2)%c) so the bound stays scalar and b
  ! is rank one; subscripting the component (x%c(2)) would retain rank one and
  ! make b rank two.
  !SYMBOLS: b {{.*}}: ObjectEntity type: REAL(4) shape: 1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x(2_8)%c,kind=8)),kind=4),kind=8)
  real :: a(x%c), b(ubound(a, 2))
  b = 0.0
end subroutine

! -fdebug-unparse-with-symbols intentionally reproduces the original bound syntax 
! rather than the synthesized rank1BoundElement node; this confirms the construct 
! still round-trips through that action with its symbol annotations.
!UNPARSE: subroutine s (n)
!UNPARSE: integer, intent(in) :: n(3)
!UNPARSE: !DEF: /s/a ObjectEntity REAL(4)
!UNPARSE: real a(n)
