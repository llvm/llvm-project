! Verify that an explicit-shape array bound which survives folding as a rank-1
! bound element round-trips through a mod file read back by a *separate*
! compilation.  Such an element arises for a rank-1 array bound that cannot be
! reduced to a plain scalar reference (here the elemental intrinsic merge() has
! a LOGICAL mask argument, which the reducer does not distribute into).  It has
! no Fortran surface syntax, so it is serialized through its UBOUND()/LBOUND()
! provenance as a real ubound(a,dim)/lbound(a,dim) call: the mod file holds that
! intrinsic, not the element, and reading it back simply re-analyzes an ordinary
! ubound/lbound reference.  The synthetic %ROBE debug marker must never appear
! in a mod file.
!
! The providers and consumers are compiled separately so that each USE genuinely
! reads the .mod from disk rather than reusing in-memory symbols.

! RUN: rm -rf %t && mkdir -p %t
! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/provenance.f90
! RUN: FileCheck %s --check-prefix=PROV --input-file=%t/m_provenance.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/unfolded.f90
! RUN: FileCheck %s --check-prefix=UNFOLDED --input-file=%t/m_unfolded.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/dependency.f90
! RUN: FileCheck %s --check-prefix=DEP --input-file=%t/m_dependency.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/consumer.f90

! `a` has the default lower bound 1, so UBOUND(a,dim) is exact for every
! dimension (max(0,ub) == max(0,UBOUND(a,dim)) for all ub) and the surviving
! element is stamped with ubound() provenance.  Both the direct ubound(a,2) and
! the size(a,2) that folds to it serialize as that intrinsic.
!--- provenance.f90
module m_provenance
contains
  subroutine s(n, m, c, a, b, d)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n,m,c))
    real :: b(ubound(a,2))
    real :: d(size(a,2))
    b(1) = 1.0
    a(1, 1) = b(1)
    d(1) = 1.0
  end subroutine
end module
! PROV-LABEL: subroutine s(n,m,c,a,b,d)
! PROV: real(4)::a(1_8:__builtin_int(merge(n,m,c),kind=8))
! PROV: real(4)::b(1_8:{{.*}}ubound(a,dim=2,kind=8){{.*}})
! PROV: real(4)::d(1_8:{{.*}}ubound(a,dim=2,kind=8){{.*}})
! PROV-NOT: {{[%][Rr][Oo][Bb][Ee]}}

! `a` has nonconstant lower bounds, so neither UBOUND nor LBOUND is exact for an
! empty dimension and no provenance can be stamped: every inquiry stays an
! unfolded runtime call rather than serializing a raw element.
!--- unfolded.f90
module m_unfolded
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
! UNFOLDED-LABEL: subroutine s(n,m,c,a,b,d)
! UNFOLDED: real(4)::b(__builtin_int(lbound(a,1_4),kind=8):__builtin_int(ubound(a,2_4),kind=8))
! UNFOLDED: real(4)::d(__builtin_int(lbound(a,2_4),kind=8):__builtin_int(size(a,2_4),kind=8))
! UNFOLDED-NOT: {{[%][Rr][Oo][Bb][Ee]}}

! A bound serialized as ubound(x,dim) names x, which the expression's own symbol
! traversal does not visit (only the element's base is traversed).  The writer
! must still declare x in the interface, ahead of the entity that inquires it:
!  - `local_shape` is a local of make_result, shadowing a module variable of
!    the same name and shape (2,9).  Omitting the local would silently bind the
!    consumer's ubound(local_shape,2) to the module variable (9 instead of 3).
!  - `z` is a dummy that follows the dependent dummy `b` in the argument list;
!    its declaration must still precede b's.
!--- dependency.f90
module m_dependency
  implicit none
  real :: local_shape(2, 9)
contains
  function make_result(n, m, mask) result(r)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: mask(2)
    real :: local_shape(merge(n, m, mask))
    real :: r(ubound(local_shape, 2))
    r = 7.0
  end function
  subroutine s(n, m, c, b, z)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: z(merge(n, m, c))
    real :: b(size(z, 2))
    b(1) = 1.0
    z(1, 1) = b(1)
  end subroutine
end module
! DEP-LABEL: function make_result(n,m,mask) result(r)
! DEP: real(4)::local_shape(1_8:__builtin_int(merge(n,m,mask),kind=8))
! DEP: real(4)::r(1_8:{{.*}}ubound(local_shape,dim=2,kind=8){{.*}})
! DEP-LABEL: subroutine s(n,m,c,b,z)
! DEP: real(4)::z(1_8:__builtin_int(merge(n,m,c),kind=8))
! DEP: real(4)::b(1_8:{{.*}}ubound(z,dim=2,kind=8){{.*}})
! DEP-NOT: {{[%][Rr][Oo][Bb][Ee]}}

!--- consumer.f90
program p
  use m_provenance, only: s1 => s
  use m_unfolded, only: s2 => s
  use m_dependency, only: make_result, s3 => s
  real, allocatable :: x(:)
  x = make_result([2, 3], [4, 5], [.true., .true.])
end program
