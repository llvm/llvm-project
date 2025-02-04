! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  type boring
  end type
  type hasAlloc
    real, allocatable :: x
  end type
  type hasInit
    real :: x = 1.
  end type
  type hasFinal
   contains
    final final
  end type
 contains
  elemental subroutine final(x)
    type(hasFinal), intent(in out) :: x
  end

  recursive subroutine typeOutAssumedRank(a,b,c,d)
    type(boring), intent(out) :: a(..)
    type(hasAlloc), intent(out) :: b(..)
    type(hasInit), intent(out) :: c(..)
    type(hasFinal), intent(out) :: d(..)
    !PORTABILITY: Assumed-rank actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call typeOutAssumedRank(a, b, c, d)
    !PORTABILITY: Assumed-rank actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call classOutAssumedRank(a, b, c, d)
    !PORTABILITY: Assumed-rank actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a, b, c, d)
  end
  recursive subroutine typeOutAssumedRankAlloc(a,b,c,d)
    type(boring), intent(out), allocatable :: a(..)
    type(hasAlloc), intent(out), allocatable :: b(..)
    type(hasInit), intent(out), allocatable :: c(..)
    type(hasFinal), intent(out), allocatable :: d(..)
    call typeOutAssumedRank(a, b, c, d)
    call typeOutAssumedRankAlloc(a, b, c, d)
  end
  recursive subroutine classOutAssumedRank(a,b,c,d)
    class(boring), intent(out) :: a(..)
    class(hasAlloc), intent(out) :: b(..)
    class(hasInit), intent(out) :: c(..)
    class(hasFinal), intent(out) :: d(..)
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call typeOutAssumedRank(a, b, c, d)
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call classOutAssumedRank(a, b, c, d)
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a, b, c, d)
  end
  recursive subroutine classOutAssumedRankAlloc(a,b,c,d)
    class(boring), intent(out), allocatable :: a(..)
    class(hasAlloc), intent(out), allocatable :: b(..)
    class(hasInit), intent(out), allocatable :: c(..)
    class(hasFinal), intent(out), allocatable :: d(..)
    call classOutAssumedRank(a, b, c, d)
    call classOutAssumedRankAlloc(a, b, c, d)
    call unlimitedOutAssumedRank(a, b, c, d)
  end
  recursive subroutine unlimitedOutAssumedRank(a,b,c,d)
    class(*), intent(out) :: a(..), b(..), c(..), d(..)
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-rank actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a, b, c, d)
  end
  recursive subroutine unlimitedOutAssumedRankAlloc(a,b,c,d)
    class(*), intent(out), allocatable :: a(..), b(..), c(..), d(..)
    call unlimitedOutAssumedRank(a, b, c, d)
    call unlimitedOutAssumedRankAlloc(a, b, c, d)
  end

  subroutine typeAssumedSize(a,b,c,d)
    type(boring) a(*)
    type(hasAlloc) b(*)
    type(hasInit) c(*)
    type(hasFinal) d(*)
    !PORTABILITY: Assumed-size actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call typeOutAssumedRank(a,b,c,d)
    !PORTABILITY: Assumed-size actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call classOutAssumedRank(a,b,c,d)
    !PORTABILITY: Assumed-size actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a,b,c,d)
  end
  subroutine classAssumedSize(a,b,c,d)
    class(boring) a(*)
    class(hasAlloc) b(*)
    class(hasInit) c(*)
    class(hasFinal) d(*)
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call classOutAssumedRank(a,b,c,d)
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a,b,c,d)
  end
  subroutine unlimitedAssumedSize(a,b,c,d)
    class(*) a(*), b(*), c(*), d(*)
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    !ERROR: Assumed-size actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization
    call unlimitedOutAssumedRank(a, b, c, d)
  end
end
