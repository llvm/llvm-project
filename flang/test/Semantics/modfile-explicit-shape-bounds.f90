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
    real, intent(inout) :: a_bothfold(lbs_fold : ubs_fold)   ! mixed const lbs_fold and dummy n
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
! `__builtin_rank1_bound_element(...)` spelling.
!===============================================================================

!-------------------------------------------------------------------------------
! Reducible bases: the single ROBE collapses to a plain scalar element ref
! (no __builtin_rank1_bound_element in the mod file).
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
! helper can't distribute element extraction into the base, so the synthetic
! `__builtin_rank1_bound_element(...)` spelling is emitted. This is the safe
! fallback that makes the wrapper necessary -- it guarantees the mod file is
! always emittable and round-trips even when reduction isn't possible. Here the
! elemental intrinsic merge() has a LOGICAL mask argument, which the helper (it
! only recurses into integer array arguments) does not reduce.
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
!real(4)::b(1_8:__builtin_int(__builtin_int(max(0_8,__builtin_rank1_bound_element(__builtin_int(merge(n,m,c),kind=8),dim=2)),kind=4),kind=8))
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
! ROBE is ever formed (no __builtin_rank1_bound_element).
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
