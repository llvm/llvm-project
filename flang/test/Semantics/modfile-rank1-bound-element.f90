! Verify that a mod file which contains a __builtin_rank1_bound_element wrapper
! can be read back in through USE from a *separate* compilation.  The wrapper is
! emitted for a rank-1 bound element that cannot be reduced to a plain scalar
! reference (here the elemental intrinsic merge() has a LOGICAL mask argument,
! which the reducer does not distribute into).  It has no true Fortran surface
! syntax and no backing intrinsic, so reading the mod file back in must rebuild
! the RankOneBoundElement node from the wrapper call.  The wrapper name must
! also be spelled in lower case so it still matches after names are folded to
! lower case when the mod file is read back in.
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
  subroutine s(n, m, c, a, b)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: b(ubound(a, 2))
    b(1) = 1.0
    a(1, 1) = b(1)
  end subroutine
end module

!--- consumer.f90
program p
  use m_rank1_bound_element_provider
end program
