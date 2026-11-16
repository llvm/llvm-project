! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test mod-file generation for F2023 explicit-shape bounds using rank-1
! integer arrays (ExplicitShapeBoundsSpec / RankOneBoundElement).

! PARAMETER rank-1 array as upper bounds
module m1
  integer, parameter :: dims(3) = [5, 10, 15]
  real :: a(dims)
end module

!Expect: m1.mod
!module m1
!integer(4),parameter::dims(1_8:3_8)=[INTEGER(4)::5_4,10_4,15_4]
!real(4)::a(1_8:[INTEGER(8)::5_8,10_8,15_8])
!end

! Rank-1 dummy as upper bounds
module m2
contains
subroutine sub1(n,a)
  integer, intent(in) :: n(3)
  real :: a(n)
end subroutine
end module

!Expect: m2.mod
!module m2
!contains
!subroutine sub1(n,a)
!integer(4),intent(in)::n(1_8:3_8)
!real(4)::a(1_8:__builtin_int(n,kind=8))
!end
!end

! Both lower and upper rank-1 bounds
module m3
contains
subroutine sub2(lb,ub,a)
  integer, intent(in) :: lb(2), ub(2)
  real :: a(lb:ub)
end subroutine
end module

!Expect: m3.mod
!module m3
!contains
!subroutine sub2(lb,ub,a)
!integer(4),intent(in)::lb(1_8:2_8)
!integer(4),intent(in)::ub(1_8:2_8)
!real(4)::a(__builtin_int(lb,kind=8):__builtin_int(ub,kind=8))
!end
!end

! Zero-size bounds array in an entity-decl declares a scalar, overriding the
! DIMENSION attribute's array-spec.
module m4
  integer, dimension(5) :: z(1 : [integer ::])
end module

!Expect: m4.mod
!module m4
!integer(4)::z
!end

module m06
  integer, parameter :: lbs_fold(2) = [0, 3]
  integer :: lbs(2) = [0, 3]
  integer, parameter :: ubs_fold(2) = [5, 7]
contains
  subroutine s(n, a_lbfold, a, a_bothfold)
    integer, intent(in) :: n(2)
    real, intent(inout) :: a_lbfold(lbs_fold : n)   ! mixed const lbs_fold and dummy n
    real, intent(inout) :: a(lbs : n)
    real, intent(inout) :: a_bothfold(lbs_fold : ubs_fold)   ! both const parameters lbs_fold and ubs_fold
  end subroutine                      
end module

!Expect: m06.mod
!module m06
!integer(4),parameter::lbs_fold(1_8:2_8)=[INTEGER(4)::0_4,3_4]
!integer(4)::lbs(1_8:2_8)
!integer(4),parameter::ubs_fold(1_8:2_8)=[INTEGER(4)::5_4,7_4]
!contains
!subroutines(n,a_lbfold,a,a_bothfold)
!integer(4),intent(in)::n(1_8:2_8)
!real(4),intent(inout)::a_lbfold([INTEGER(8)::0_8,3_8]:__builtin_int(n,kind=8))
!real(4),intent(inout)::a(__builtin_int(lbs,kind=8):__builtin_int(n,kind=8))
!real(4),intent(inout)::a_bothfold([INTEGER(8)::0_8,3_8]:[INTEGER(8)::5_8,7_8])
!end
!end

module ccm1
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(2)
    real :: a(n)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: ccm1.mod
! module ccm1
! contains
! subroutine s(n,a,b)
! integer(4),intent(in)::n(1_8:2_8)
! real(4)::a(1_8:__builtin_int(n,kind=8))
! real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8))
! end
! end

!===============================================================================
! SINGLE RankOneBoundElement (ROBE) reduction
!
! In every module below, `a` is given a rank-1 integer base of extent 2, so it
! is a rank-2 F2023 explicit-shape array (one bounds-spec per element of the
! base). `b(ubound(a,2))` then asks for ONE dimension's
! extent: ubound(a,2) folds to a SINGLE ROBE that extracts element [2] of the
! rank-1 base. That is exactly the case we care about here -- a lone ROBE that
! must be rendered into the mod file. The FoldOperation(RankOneBoundElement)
! helper tries to reduce that ROBE to an ordinary scalar element reference so
! the mod file round-trips as valid Fortran instead of leaking the synthetic
! `%ROBE(...)` spelling.
!===============================================================================

!-------------------------------------------------------------------------------
! Reducible bases: the single ROBE collapses to a plain scalar element ref
! (no %ROBE in the mod file).
!-------------------------------------------------------------------------------

! Whole array COMPONENT base -> reduces to x%c(2_8)
module mcomp
  type t
    integer :: c(2)
  end type
contains
  subroutine s(x, a, b)
    type(t), intent(in) :: x
    real :: a(x%c)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mcomp.mod
!module mcomp
!type::t
!integer(4)::c(1_8:2_8)
!end type
!contains
!subroutine s(x,a,b)
!type(t),intent(in)::x
!real(4)::a(1_8:__builtin_int(x%c,kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x%c(2_8),kind=8)),kind=4),kind=8))
!end
!end

! Scalar COMPONENT of an array parent base -> the element subscript lands on the
! parent, reducing to x(2_8)%c (not the rank-retaining x%c(2_8))
module mcompparent
  type t
    integer :: c
  end type
contains
  subroutine s(x, a, b)
    type(t), intent(in) :: x(2)
    real :: a(x%c)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mcompparent.mod
!module mcompparent
!type::t
!integer(4)::c
!end type
!contains
!subroutine s(x,a,b)
!type(t),intent(in)::x(1_8:2_8)
!real(4)::a(1_8:__builtin_int(x%c,kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x(2_8)%c,kind=8)),kind=4),kind=8))
!end
!end

! Two part-refs, one scalar-subscripted so exactly one has nonzero rank (C919),
! reduced from either side to x(3_8)%c(2_8). a2's array part is the component c,
! so ubound(a2,2) subscripts it; a1's array part is the parent x, reached
! through the component's all-scalar ArrayRef subscripts.
module mcompboth
  type t
    integer :: c(2)
  end type
contains
  subroutine s(x, a1, a2, b1, b2)
    type(t), intent(in) :: x(3)
    real :: a1(x%c(2))
    real :: a2(x(3)%c)
    real :: b1(ubound(a1,3))
    real :: b2(ubound(a2,2))
    b1(1) = 1.0
    b2(1) = 1.0
    a1(1,1,1) = b1(1)
    a2(1,1) = b2(1)
  end subroutine
end module

!Expect: mcompboth.mod
!module mcompboth
!type::t
!integer(4)::c(1_8:2_8)
!end type
!contains
!subroutine s(x,a1,a2,b1,b2)
!type(t),intent(in)::x(1_8:3_8)
!real(4)::a1(1_8:__builtin_int(x%c(2_8),kind=8))
!real(4)::a2(1_8:__builtin_int(x(3_8)%c,kind=8))
!real(4)::b1(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x(3_8)%c(2_8),kind=8)),kind=4),kind=8))
!real(4)::b2(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x(3_8)%c(2_8),kind=8)),kind=4),kind=8))
!end
!end

! Arbitrary DataRef nesting: the single array part-ref (unsubscripted c(3),
! per C919 the only nonzero-rank part-ref) is subscripted at element [dim],
! scalar components/subscripts above and below it are rebuilt, reducing
! ubound(a,2) to x(1_8)%c(2_8)%d(3_8)%e.
module mnested
  type t3
    integer :: e
  end type
  type t2
    type(t3) :: d(4)
  end type
  type t1
    type(t2) :: c(3)
  end type
contains
  subroutine s(x, a, b)
    type(t1), intent(in) :: x(2)
    real :: a(x(1)%c%d(3)%e)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1,1) = b(1)
  end subroutine
end module

!Expect: mnested.mod
!module mnested
!type::t3
!integer(4)::e
!end type
!type::t2
!type(t3)::d(1_8:4_8)
!end type
!type::t1
!type(t2)::c(1_8:3_8)
!end type
!contains
!subroutine s(x,a,b)
!type(t1),intent(in)::x(1_8:2_8)
!real(4)::a(1_8:__builtin_int(x(1_8)%c%d(3_8)%e,kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(x(1_8)%c(2_8)%d(3_8)%e,kind=8)),kind=4),kind=8))
!end
!end

! Named-constant/PARAMETER array base -> extent folds to a constant
module mparam
  integer, parameter :: dims(2) = [5, 10]
contains
  subroutine s(a, b)
    real :: a(dims)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mparam.mod
!module mparam
!integer(4),parameter::dims(1_8:2_8)=[INTEGER(4)::5_4,10_4]
!contains
!subroutine s(a,b)
!real(4)::a(1_8:[INTEGER(8)::5_8,10_8])
!real(4)::b(1_8:10_8)
!end
!end

! Elementwise arithmetic base n+1 -> reduces to n(2_8)+1_4
module marith_add
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(2)
    real :: a(n+1)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: marith_add.mod
!module marith_add
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:__builtin_int(n+1_4,kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8)+1_4,kind=8)),kind=4),kind=8))
!end
!end

! Elementwise arithmetic base 2*n -> reduces to 2_4*n(2_8)
module marith_mul
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(2)
    real :: a(2*n)
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: marith_mul.mod
!module marith_mul
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:__builtin_int(2_4*n,kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(2_4*n(2_8),kind=8)),kind=4),kind=8))
!end
!end

! Explicit kind conversion base int(n,4) -> reduces through the Convert
module mkindconv
contains
  subroutine s(n, a, b)
    integer(8), intent(in) :: n(2)
    real :: a(int(n,4))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mkindconv.mod
!module mkindconv
!contains
!subroutine s(n,a,b)
!integer(8),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:__builtin_int(__builtin_int(n,kind=4),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(__builtin_int(n(2_8),kind=4),kind=8)),kind=4),kind=8))
!end
!end

! Array SECTION base n(1:2) -> reduces to n(2_8)
module msection
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(4)
    real :: a(n(1:2))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: msection.mod
!module msection
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:4_8)
!real(4)::a(1_8:__builtin_int(n(1_8:2_8:1_8),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8))
!end
!end

! Vector-subscripted array base n(idx) -> element [dim] of n(idx) is
! n(idx(dim)), so the vector subscript's own element reduction is substituted
! back in, reducing to n(idx(2_8))
module mvecsub
contains
  subroutine s(n, idx, a, b)
    integer, intent(in) :: n(4)
    integer, intent(in) :: idx(2)
    real :: a(n(idx))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mvecsub.mod
!module mvecsub
!contains
!subroutine s(n,idx,a,b)
!integer(4),intent(in)::n(1_8:4_8)
!integer(4),intent(in)::idx(1_8:2_8)
!real(4)::a(1_8:__builtin_int(n(__builtin_int(idx,kind=8)),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(__builtin_int(idx(2_8),kind=8)),kind=8)),kind=4),kind=8))
!end
!end

! Vector subscript over an elementwise expression n(idx+1) -> the inner index
! reduces through the full element reducer (idx+1 -> idx(2_8)+1_4), giving
! n(idx(2_8)+1_4)
module mvecsubexpr
contains
  subroutine s(n, idx, a, b)
    integer, intent(in) :: n(4)
    integer, intent(in) :: idx(2)
    real :: a(n(idx + 1))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mvecsubexpr.mod
!module mvecsubexpr
!contains
!subroutine s(n,idx,a,b)
!integer(4),intent(in)::n(1_8:4_8)
!integer(4),intent(in)::idx(1_8:2_8)
!real(4)::a(1_8:__builtin_int(n(__builtin_int(idx+1_4,kind=8)),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(__builtin_int(idx(2_8)+1_4,kind=8)),kind=8)),kind=4),kind=8))
!end
!end

! Array CONSTRUCTOR base [n(1),n(2)] -> reduces to n(2_8)
module marrcons
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(2)
    real :: a([n(1), n(2)])
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: marrcons.mod
!module marrcons
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:[INTEGER(8)::__builtin_int(n(1_8),kind=8),__builtin_int(n(2_8),kind=8)])
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8))
!end
!end

! Elemental INTRINSIC base abs(n) -> reduces to abs(n(2_8))
module melemintrin
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(2)
    real :: a(abs(n))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: melemintrin.mod
!module melemintrin
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:__builtin_int(abs(n),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(abs(n(2_8)),kind=8)),kind=4),kind=8))
!end
!end

! Array CONSTRUCTOR with an IMPLIED-DO base [(n(i),i=1,2)] -> reduces to n(2_8)
module mimplieddo
contains
  subroutine s(n, a, b)
    integer, intent(in) :: n(4)
    real :: a([(n(i), i=1,2)])
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mimplieddo.mod
!module mimplieddo
!contains
!subroutine s(n,a,b)
!integer(4),intent(in)::n(1_8:4_8)
!real(4)::a(1_8:__builtin_int([INTEGER(4)::(n(__builtin_int(__builtin_int(i,kind=4),kind=8)),INTEGER(8)::i=1_8,2_8,1_8)],kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8))
!end
!end

!-------------------------------------------------------------------------------
! Not-yet-reduced bases that keep the ROBE wrapper: a ROBE IS formed, but the
! helper can't distribute element extraction into the base, so the element
! survives folding.  When the elemet came from an UBOUND()/LBOUND() inquiry of
! another explicit-shape array (here `ubound(a,2)`), it carries provenance and
! is serialized as that real intrinsic -- `ubound(a,dim=2,kind=8)` -- which
! round-trips without the synthetic  spelling.
! Here the elemental intrinsic merge() has a LOGICAL mask argument, which the
! helper (it only recurses into integer array arguments) does not reduce.
!-------------------------------------------------------------------------------

! Elemental intrinsic with a non-integer array arg -> wrapper is kept
module mmergefallback
contains
  subroutine s(n, m, c, a, b)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mmergefallback.mod
!module mmergefallback
!contains
!subroutine s(n,m,c,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::a(1_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=2,kind=8)),kind=4),kind=8))
!end
!end

! The same irreducible base with an explicit all-ones rank-1 lower bound.  The
! constant lower bound is folded into per-dimension constants at declaration,
! so the extent of b is max(0,ub) exactly as for b(1:merge(...)) and size(b,2)
! is stamped with ubound() provenance and serialized as the real intrinsic.
! (Were the lower bound left as an unfolded element, the lower bound would not
! be recognized as 1, no provenance would be stamped, and e would serialize as
! the intact size(b,2_4) call.)
module mmergeones
contains
  subroutine s(n, m, c, b, e)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: b([1, 1] : merge(n, m, c))
    real :: e(size(b, 2))
    e(1) = 1.0
    b(1,1) = e(1)
  end subroutine
end module

!Expect: mmergeones.mod
!module mmergeones
!contains
!subroutine s(n,m,c,b,e)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::b([INTEGER(8)::1_8,1_8]:__builtin_int(merge(n,m,c),kind=8))
!real(4)::e(1_8:__builtin_int(__builtin_int(max(0_8,ubound(b,dim=2,kind=8)),kind=4),kind=8))
!end
!end

! SIZE(a,dim) on a dimension with the default lower bound of 1 folds to the same
! provenance ROBE as UBOUND(a,dim) (size == ubound when the lower bound is 1), so
! it serializes as max(0,ubound(a,dim)...) and lowers to the shared bound
! arithmetic rather than a runtime SIZE inquiry -- never the synthetic %ROBE
! spelling.
module msizeprovenance
contains
  subroutine s(n, m, c, a, b)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: b(size(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: msizeprovenance.mod
!module msizeprovenance
!contains
!subroutine s(n,m,c,a,b)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::a(1_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=2,kind=8)),kind=4),kind=8))
!end
!end

! SIZE(a,dim) on a dimension with a NON-constant lower bound cannot fold safely:
! the extent max(0,ub-lb+1) would mix a raw upper bound with an intrinsic
! lbound(a,dim), which fails to round-trip for an empty dimension (LBOUND is 1
! for an empty dimension).  GetLBOUND bails for such a bound, so SIZE keeps the
! inquiry intact and serializes as size(a,dim) -- mirroring UBOUND(), which also
! declines here.
module msizenonconstlower
contains
  subroutine s(lo, m, c, hi, a, b)
    integer, intent(in) :: lo(2), m(2), hi(2)
    logical, intent(in) :: c(2)
    real :: a(merge(lo, m, c) : hi)
    real :: b(size(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: msizenonconstlower.mod
!module msizenonconstlower
!contains
!subroutine s(lo,m,c,hi,a,b)
!integer(4),intent(in)::lo(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!integer(4),intent(in)::hi(1_8:2_8)
!real(4)::a(__builtin_int(merge(lo,m,c),kind=8):__builtin_int(hi,kind=8))
!real(4)::b(1_8:__builtin_int(size(a,2_4),kind=8))
!end
!end

!-------------------------------------------------------------------------------
! Fundamentally irreducible bases: there is NO Fortran syntax to subscript a
! single element of the base -- e.g. you cannot index a function result (gb()(2)
! is not legal) -- so a ROBE over such a base could never be rendered as a
! scalar element ref, even though the base itself is legal in a declaration.
! Here ubound side-steps the problem entirely by folding to size(a,dim=2), so no
! ROBE is ever formed.
!-------------------------------------------------------------------------------

! User/specification FUNCTION base -> ubound degrades to size(a,dim=2), so no
! ROBE is ever formed (no %ROBE).
module muserfunc
contains
  pure function gb() result(r)
    integer :: r(2)
    r = [3, 4]
  end function
  subroutine s(a, b)
    real :: a(gb())
    real :: b(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: muserfunc.mod
!module muserfunc
!contains
!pure function gb() result(r)
!integer(4)::r(1_8:2_8)
!end
!subroutine s(a,b)
!real(4)::a(1_8:__builtin_int(gb(),kind=8))
!real(4)::b(1_8:size(a,dim=2,kind=8))
!end
!end

!===============================================================================
! LBOUND emission
!
! Unlike UBOUND, an LBOUND(a,dim) inquiry on a dimension whose declared lower
! bound is a non-constant rank-1 bound expression does NOT fold to a provenance
! ROBE.  LBOUND is defined to return 1 for an empty dimension, so the folder
! only replaces it with the declared lower bound when it can prove the dimension
! is nonempty.  It cannot prove that for a non-constant bound, so it leaves the
! inquiry intact and serializes it as lbound(a,dim) -- mirroring the size(a,dim)
! case, never the %ROBE spelling.
!===============================================================================

! Plain rank-1 array lower bound -> lbound(a,2) stays unfolded.
module mlboundplain
contains
  subroutine s(lo, hi, a, b)
    integer, intent(in) :: lo(2), hi(2)
    real :: a(lo : hi)
    real :: b(lbound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mlboundplain.mod
!module mlboundplain
!contains
!subroutine s(lo,hi,a,b)
!integer(4),intent(in)::lo(1_8:2_8)
!integer(4),intent(in)::hi(1_8:2_8)
!real(4)::a(__builtin_int(lo,kind=8):__builtin_int(hi,kind=8))
!real(4)::b(1_8:__builtin_int(lbound(a,2_4),kind=8))
!end
!end

! Surviving (merge) rank-1 lower base -> lbound(a,2) still stays unfolded.
module mlboundmerge
contains
  subroutine s(lo, m, c, hi, a, b)
    integer, intent(in) :: lo(2), m(2), hi(2)
    logical, intent(in) :: c(2)
    real :: a(merge(lo, m, c) : hi)
    real :: b(lbound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module

!Expect: mlboundmerge.mod
!module mlboundmerge
!contains
!subroutine s(lo,m,c,hi,a,b)
!integer(4),intent(in)::lo(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!integer(4),intent(in)::hi(1_8:2_8)
!real(4)::a(__builtin_int(merge(lo,m,c),kind=8):__builtin_int(hi,kind=8))
!real(4)::b(1_8:__builtin_int(lbound(a,2_4),kind=8))
!end
!end

!===============================================================================
! Inquiry functions WITHOUT a DIM argument
!
! ubound(a)/lbound(a) with no DIM are themselves rank-1 integer arrays, so they
! are F2023 explicit-shape bases in their own right: b and d become rank-2
! arrays, one bounds-spec per element.  These are declaration-time bounds (the
! element carries no provenance), so the writer unwraps the whole rank-1 base
! and serializes the folded array literal directly -- no ROBE is formed and no
! %ROBE spelling can leak.  size(a) with no DIM is a scalar, so e is an ordinary
! rank-1 array with a plain scalar bound.
!===============================================================================
module mnodim
contains
  subroutine s(n, a, b, d, e)
    integer, intent(in) :: n(2)
    real :: a(n)
    real :: b(ubound(a))
    real :: d(lbound(a) : ubound(a))
    real :: e(size(a))
    b(1,1) = 1.0
    a(1,1) = b(1,1)
  end subroutine
end module

!Expect: mnodim.mod
!module mnodim
!contains
!subroutine s(n,a,b,d,e)
!integer(4),intent(in)::n(1_8:2_8)
!real(4)::a(1_8:__builtin_int(n,kind=8))
!real(4)::b(1_8:[INTEGER(8)::__builtin_int(__builtin_int(max(0_8,__builtin_int(n(1_8),kind=8)),kind=4),kind=8),__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8)])
!real(4)::d([INTEGER(8)::1_8,1_8]:[INTEGER(8)::__builtin_int(__builtin_int(max(0_8,__builtin_int(n(1_8),kind=8)),kind=4),kind=8),__builtin_int(__builtin_int(max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8)])
!real(4)::e(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_int(n(1_8),kind=8))*max(0_8,__builtin_int(n(2_8),kind=8)),kind=4),kind=8))
!end
!end

!===============================================================================
! Extent-aggregating inquiries over an irreducible (merge) base
!
! shape(a), size(a) with no DIM, and sizeof(a) all expand every dimension to its
! extent and aggregate them (a rank-1 vector, a product, and a byte count).  When
! the base cannot be reduced to a single element (here the merge() makes it
! irreducible), the surviving upper-bound element has no Fortran surface syntax
! and its only debug rendering is the %ROBE marker, which must never reach a mod
! file.  With the default lower bound of 1 the extent is max(0, ub), which equals
! max(0, UBOUND(a,dim)) even for an empty dimension, so the element is stamped
! with ubound(a,dim) provenance when it is copied out of a's shape and all three
! inquiries -- plus size(a,2) with DIM -- serialize as real ubound() intrinsics
! and round-trip.  This is the regression guard for the aggregate-inquiry leak.
!===============================================================================
module maggregprovenance
contains
  subroutine s(n, m, c, a, bshape, bsizedim, bsize, bbytes)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: bshape(shape(a))
    real :: bsizedim(size(a, 2))
    real :: bsize(size(a))
    real :: bbytes(sizeof(a))
    bshape(1,1) = 1.0
    bsizedim(1) = 1.0
    bsize(1) = 1.0
    bbytes(1) = 1.0
    a(1,1) = bshape(1,1)
  end subroutine
end module

!Expect: maggregprovenance.mod
!module maggregprovenance
!contains
!subroutine s(n,m,c,a,bshape,bsizedim,bsize,bbytes)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::a(1_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::bshape(1_8:[INTEGER(8)::__builtin_int(__builtin_int(max(0_8,ubound(a,dim=1,kind=8)),kind=4),kind=8),__builtin_int(__builtin_int(max(0_8,ubound(a,dim=2,kind=8)),kind=4),kind=8)])
!real(4)::bsizedim(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=2,kind=8)),kind=4),kind=8))
!real(4)::bsize(1_8:__builtin_int(__builtin_int(max(0_8,ubound(a,dim=1,kind=8))*max(0_8,ubound(a,dim=2,kind=8)),kind=4),kind=8))
!real(4)::bbytes(1_8:4_8*max(0_8,ubound(a,dim=1,kind=8))*max(0_8,ubound(a,dim=2,kind=8)))
!end
!end

!===============================================================================
! Extent-aggregating inquiries over a non-constant (merge) LOWER bound
!
! Here the lower bound is itself an irreducible rank-1 element.  The extent
! max(0, ub - lb + 1) embeds it raw: LBOUND(a,dim) is 1, not the declared lower
! bound, for an empty dimension, so lbound() provenance would be inexact and the
! element has no Fortran spelling at all.  Rather than emit the non-Fortran %ROBE
! marker, shape(a)/size(a)/sizeof(a) keep themselves intact and serialize as the
! real transformational inquiries, which round-trip as legal Fortran.
!===============================================================================
module maggregkeep
contains
  subroutine s(lo, m, c, hi, a, bshape, bsize, bbytes)
    integer, intent(in) :: lo(2), m(2), hi(2)
    logical, intent(in) :: c(2)
    real :: a(merge(lo, m, c) : hi)
    real :: bshape(shape(a))
    real :: bsize(size(a))
    real :: bbytes(sizeof(a))
    bshape(1,1) = 1.0
    bsize(1) = 1.0
    bbytes(1) = 1.0
    a(1,1) = bshape(1,1)
  end subroutine
end module

!Expect: maggregkeep.mod
!module maggregkeep
!contains
!subroutine s(lo,m,c,hi,a,bshape,bsize,bbytes)
!integer(4),intent(in)::lo(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!integer(4),intent(in)::hi(1_8:2_8)
!real(4)::a(__builtin_int(merge(lo,m,c),kind=8):__builtin_int(hi,kind=8))
!real(4)::bshape(1_8:__builtin_int(shape(a),kind=8))
!real(4)::bsize(1_8:__builtin_int(size(a),kind=8))
!real(4)::bbytes(1_8:sizeof(a))
!end
!end

!===============================================================================
! Extent-aggregating inquiries over a constant lower bound other than 1
!
! ubound(a,dim) provenance is exact only when the lower bound is 1: with a lower
! bound of 0 the extent is max(0, ub + 1), but UBOUND(a,dim) of an empty
! dimension is 0, so max(0, ubound(a,dim) + 1) would be 1 instead of 0.  The
! element is therefore left raw and every inquiry -- including size(a,2) with
! DIM -- declines to fold rather than serialize a semantically wrong bound.
!===============================================================================
module maggregzerolower
contains
  subroutine s(n, m, c, a, bshape, bsizedim, bsize, bbytes)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(0 : merge(n, m, c))
    real :: bshape(shape(a))
    real :: bsizedim(size(a, 2))
    real :: bsize(size(a))
    real :: bbytes(sizeof(a))
    bshape(1,1) = 1.0
    bsizedim(1) = 1.0
    bsize(1) = 1.0
    bbytes(1) = 1.0
    a(1,1) = bshape(1,1)
  end subroutine
end module

!Expect: maggregzerolower.mod
!module maggregzerolower
!contains
!subroutine s(n,m,c,a,bshape,bsizedim,bsize,bbytes)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::a(0_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::bshape(1_8:__builtin_int(shape(a),kind=8))
!real(4)::bsizedim(1_8:__builtin_int(size(a,2_4),kind=8))
!real(4)::bsize(1_8:__builtin_int(size(a),kind=8))
!real(4)::bbytes(1_8:sizeof(a))
!end
!end

!===============================================================================
! UBOUND through the shape-derived path
!
! ubound(transpose(a),DIM), ubound((a)) and ubound(x%a) take their bounds from
! the operand's shape rather than from a named array's declared bounds, so a
! raw rank-1 bound element in an extent (a has a lower bound of 0, making
! neither UBOUND nor LBOUND exact) must keep the inquiry from folding exactly as
! it keeps size()/shape() from folding: the call is kept intact and serialized
! as a runtime inquiry, never as a raw element.  With DIM the result is a plain
! scalar bound; without DIM it is itself a rank-1 explicit-shape base.
!===============================================================================
module mshapepath
contains
  subroutine s(n, m, c, a, b, d)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(0 : merge(n, m, c))
    real :: b(ubound(transpose(a), 1))
    real :: d(ubound((a)))
    b(1) = 1.0
    d(1, 1) = 1.0
    a(1, 1) = b(1)
  end subroutine
end module

!Expect: mshapepath.mod
!module mshapepath
!contains
!subroutine s(n,m,c,a,b,d)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::a(0_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::b(1_8:__builtin_int(ubound(transpose(a),1_4),kind=8))
!real(4)::d(1_8:__builtin_int(ubound((a)),kind=8))
!end
!end

!===============================================================================
! A character length through the same path
!
! The writer checks every expression it prints, not only array bounds: a raw
! element reaching a character length would be serialized as the unparsable
! %ROBE marker.  The inquiry must stay intact here too.
!===============================================================================
module mcharlen
contains
  subroutine s(l, u, a, c)
    integer, intent(in) :: l(2), u(2)
    real, intent(inout) :: a(merge(l, l, [.true., .true.]) : merge(u, u, [.true., .true.]))
    character(len=ubound((a), 2)), intent(inout) :: c
  end subroutine
end module

!Expect: mcharlen.mod
!module mcharlen
!contains
!subroutine s(l,u,a,c)
!integer(4),intent(in)::l(1_8:2_8)
!integer(4),intent(in)::u(1_8:2_8)
!real(4),intent(inout)::a(__builtin_int(merge(l,l,[LOGICAL(4)::.true._4,.true._4]),kind=8):__builtin_int(merge(u,u,[LOGICAL(4)::.true._4,.true._4]),kind=8))
!character(ubound((a),2_4),1),intent(inout)::c
!end
!end

!===============================================================================
! A constant base is folded at declaration
!
! The constant bounds array [2,3] is folded into per-dimension constant bounds
! when a is declared, so no raw element is left in any extent even though the
! lower bound 0 would make UBOUND inexact: size(a) is the constant 3*4 == 12.
! The mod file still writes the upper bound as the declared rank-1 array.
!===============================================================================
module msizeconst
  real :: a(0 : [2, 3])
  integer, parameter :: count = size(a)
end module

!Expect: msizeconst.mod
!module msizeconst
!real(4)::a(0_8:[INTEGER(8)::2_8,3_8])
!integer(4),parameter::count=12_4
!intrinsic::size
!end

!===============================================================================
! Provenance declares its originating array
!
! A bound serialized as ubound(z,dim) names z, which the expression's symbol
! traversal does not visit; the writer declares z ahead of the dependent entity
! even when z follows it in the dummy argument list.
!===============================================================================
module mdeporder
contains
  subroutine s(n, m, c, b, z)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: z(merge(n, m, c))
    real :: b(size(z, 2))
    b(1) = 1.0
    z(1, 1) = b(1)
  end subroutine
end module

!Expect: mdeporder.mod
!module mdeporder
!contains
!subroutine s(n,m,c,b,z)
!integer(4),intent(in)::n(1_8:2_8)
!integer(4),intent(in)::m(1_8:2_8)
!logical(4),intent(in)::c(1_8:2_8)
!real(4)::z(1_8:__builtin_int(merge(n,m,c),kind=8))
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,ubound(z,dim=2,kind=8)),kind=4),kind=8))
!end
!end
