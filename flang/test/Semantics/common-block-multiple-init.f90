! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

! A *duplicate* (identical-valued) initialization of a named COMMON block
! across program units is accepted as a nonstandard extension with a
! portability warning; a genuinely *conflicting* one is still a hard
! error. This covers the various shapes both checks apply to, and pins
! that -pedantic does not change the accepted case's severity (unlike
! most other checks gated by -pedantic, this one is not tied to strict
! standard conformance).

! Control: two DATA statements in the same program unit, each initializing
! a distinct member of the same COMMON block, is a single (first)
! appearance -- not "multiple initialization".
subroutine same_unit_control
  integer :: p, q
  common /cs/ p, q
  data p /1/
  data q /2/
end subroutine

!-------------------------------------------------------------------------
! Accepted: duplicate (identical-valued) initialization.
!-------------------------------------------------------------------------

! Baseline: DATA-statement initialization of the same block, with the same
! value, in two program units.
subroutine data_dup_first
  integer :: i
  common /cd/ i
  data i /111/
end subroutine
subroutine data_dup_second
  !PORTABILITY: Multiple initialization of COMMON block /cd/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /cd/ i
  data i /111/
end subroutine

! The check also applies to declaration initializers, not just DATA
! statements.
subroutine decl_dup_first
  integer :: i = 111
  common /ce/ i
end subroutine
subroutine decl_dup_second
  !PORTABILITY: Multiple initialization of COMMON block /ce/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i = 111
  common /ce/ i
end subroutine

! Mixed: one appearance uses a declaration initializer, the other a DATA
! statement, both with the same value.
subroutine mixed_dup_decl
  integer :: i = 111
  common /cf/ i
end subroutine
subroutine mixed_dup_data
  !PORTABILITY: Multiple initialization of COMMON block /cf/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /cf/ i
  data i /111/
end subroutine

! Three duplicate appearances: the second and third each warn, both
! against the first.
subroutine dup_three_a
  integer :: i
  common /cg/ i
  data i /1/
end subroutine
subroutine dup_three_b
  !PORTABILITY: Multiple initialization of COMMON block /cg/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /cg/ i
  data i /1/
end subroutine
subroutine dup_three_c
  !PORTABILITY: Multiple initialization of COMMON block /cg/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /cg/ i
  data i /1/
end subroutine

! An uninitialized first appearance is not "the first appearance" for this
! check -- only the first *initialized* appearance matters, and later
! initialized appearances are compared against it, not against the
! uninitialized one.
subroutine dup_uninit_first
  integer :: i
  common /ch/ i
end subroutine
subroutine dup_uninit_second
  integer :: i
  common /ch/ i
  data i /1/
end subroutine
subroutine dup_uninit_third
  !PORTABILITY: Multiple initialization of COMMON block /ch/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /ch/ i
  data i /1/
end subroutine

! An array member initialized to the same constant in both appearances.
subroutine array_dup_first
  integer :: a(3)
  common /ci/ a
  data a /1, 2, 3/
end subroutine
subroutine array_dup_second
  !PORTABILITY: Multiple initialization of COMMON block /ci/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: a(3)
  common /ci/ a
  data a /1, 2, 3/
end subroutine

! An array member initialized with a scalar in both appearances.
subroutine array_scalar_dup_first
  real :: x(10) = 1.0
  common /cj/ x
end subroutine
subroutine array_scalar_dup_second
  !PORTABILITY: Multiple initialization of COMMON block /cj/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  real :: x(10) = 1.0
  common /cj/ x
end subroutine

! An array member initialized with a scalar in one appearance and with the
! same values by a DATA statement in the other, in either order.
subroutine array_scalar_data_first
  real :: x(10) = 1.0
  common /ck/ x
end subroutine
subroutine array_scalar_data_second
  !PORTABILITY: Multiple initialization of COMMON block /ck/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  real :: x(10)
  common /ck/ x
  data x /10*1.0/
end subroutine
subroutine array_data_scalar_first
  integer :: a(0:2)
  common /cl/ a
  data a /3*7/
end subroutine
subroutine array_data_scalar_second
  !PORTABILITY: Multiple initialization of COMMON block /cl/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: a(0:2) = 7
  common /cl/ a
end subroutine

!-------------------------------------------------------------------------
! Rejected: genuinely conflicting initialization.
!-------------------------------------------------------------------------

! DATA-statement initialization of the same member with different values.
subroutine data_conflict_first
  integer :: i
  common /da/ i
  data i /111/
end subroutine
subroutine data_conflict_second
  !ERROR: Multiple initialization of COMMON block /da/
  integer :: i
  common /da/ i
  data i /222/
end subroutine

! Declaration initializers with different values.
subroutine decl_conflict_first
  integer :: i = 111
  common /db/ i
end subroutine
subroutine decl_conflict_second
  !ERROR: Multiple initialization of COMMON block /db/
  integer :: i = 222
  common /db/ i
end subroutine

! Disjoint members: each appearance initializes a *different* member of the
! same block. This is not a duplicate initialization (neither appearance
! repeats the other's value), so it is rejected rather than accepted, even
! though the two appearances do not directly disagree on any one member's
! value.
subroutine disjoint_first
  integer :: i, j
  common /dc/ i, j
  data i /111/
end subroutine
subroutine disjoint_second
  !ERROR: Multiple initialization of COMMON block /dc/
  integer :: i, j
  common /dc/ i, j
  data j /222/
end subroutine

! Three appearances where only one conflicts: the third disagrees with the
! (duplicate) first and second, and is rejected; the second still matches
! the first and is accepted.
subroutine three_mixed_a
  integer :: i
  common /dd/ i
  data i /1/
end subroutine
subroutine three_mixed_b
  !PORTABILITY: Multiple initialization of COMMON block /dd/ is not standard; this appearance duplicates the previous initialization [-Wmultiple-common-block-init]
  integer :: i
  common /dd/ i
  data i /1/
end subroutine
subroutine three_mixed_c
  !ERROR: Multiple initialization of COMMON block /dd/
  integer :: i
  common /dd/ i
  data i /2/
end subroutine

! A COMMON block member that is never itself directly initialized (no DATA
! statement, no declaration initializer) but is indirectly initialized via
! an equivalenced object is conservatively rejected as a conflict, rather
! than being compared for a duplicate value: the equivalenced objects may
! differ in name, offset, or type across appearances, so no attempt is
! made to determine whether they truly agree.
subroutine equiv_first
  integer :: i, e
  common /de/ i
  equivalence (i, e)
  data e /111/
end subroutine
subroutine equiv_second
  !ERROR: Multiple initialization of COMMON block /de/
  integer :: i, e
  common /de/ i
  equivalence (i, e)
  data e /111/
end subroutine

! +0.0 and -0.0 compare equal under Fortran's == operator, but they are not
! the same value (they have different representations), so this is a
! conflict, not a duplicate.
subroutine zero_first
  real :: x
  common /df/ x
  data x /0.0/
end subroutine
subroutine zero_second
  !ERROR: Multiple initialization of COMMON block /df/
  real :: x
  common /df/ x
  data x /-0.0/
end subroutine

! An array member initialized with a scalar in one appearance and by a DATA
! statement that differs in one element in the other.
subroutine array_scalar_conflict_first
  real :: x(3) = 1.0
  common /dg/ x
end subroutine
subroutine array_scalar_conflict_second
  !ERROR: Multiple initialization of COMMON block /dg/
  real :: x(3)
  common /dg/ x
  data x /1.0, 1.0, 2.0/
end subroutine
