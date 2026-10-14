! Verify that an explicit-shape array bound which survives folding as a rank-1
! bound element round-trips through a mod file read back by a *separate*
! compilation.  Such an element arises for a rank-1 array bound that cannot be
! reduced to a plain scalar reference (here the elemental intrinsic merge() has
! a LOGICAL mask argument, which the reducer does not distribute into).  It has
! no Fortran surface syntax, so it is serialized through its UBOUND()/LBOUND()
! provenance as a real ubound(a,dim)/lbound(a,dim) call: the mod file holds that
! intrinsic, not the element, and reading it back simply re-analyzes an ordinary
! ubound/lbound reference.  The synthetic %ROBE debug marker must never appear
! in the mod file.
!
! The provider and consumer are compiled separately so that the USE genuinely
! reads provider.mod from disk rather than reusing in-memory symbols.

! RUN: rm -rf %t && mkdir -p %t
! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/provider.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/consumer.f90

!--- provider.f90
module m_rank1_bound_element_provider
contains
  subroutine s(n, m, c, a, b, d)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n,m,c) : merge(n,m,c))
    real :: b(lbound(a,1) : ubound(a,2))
    real :: d(lbound(a,2) : size(a,2))
    b(1) = 1.0
    a(1, 1) = b(1)
  end subroutine
end module

!--- consumer.f90
program p
  use m_rank1_bound_element_provider
end program
