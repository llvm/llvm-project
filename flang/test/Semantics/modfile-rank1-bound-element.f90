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
! RUN: %flang_fc1 -fsyntax-only -J%t %t/submod_parent.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/submod_child.f90
! RUN: FileCheck %s --check-prefix=SUBMOD --input-file=%t/m_submod-child.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/submod_grandchild.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/fixedpoint_parent.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/fixedpoint_child.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/fixedpoint_consumer.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %t/reverse_provider.f90
! RUN: FileCheck %s --check-prefix=REVERSE --input-file=%t/m_reverse.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/reverse_consumer.f90
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

! A `module procedure` body in a submodule inherits its dummies by copying the
! interface's symbols and remapping every symbol reference in their expressions
! to the body's own copies.  The provenance array is outside the traversed
! expression tree, so it must be remapped explicitly; otherwise the body's `b`
! would still name the interface's `z`, the child's mod file would declare `b`
! before its own `z`, and a descendant submodule could not use the interface.
!--- submod_parent.f90
module m_submod
  interface
    module subroutine sep(n, m, c, b, z)
      integer, intent(in) :: n(2), m(2)
      logical, intent(in) :: c(2)
      real :: z(merge(n, m, c))
      real :: b(size(z, 2))
    end subroutine
    module subroutine report()
    end subroutine
  end interface
end module

!--- submod_child.f90
submodule(m_submod) child
contains
  module procedure sep
    b = 1.0
    z = 2.0
  end procedure
end submodule
! SUBMOD-LABEL: module subroutine sep(n,m,c,b,z)
! SUBMOD: real(4)::z(1_8:__builtin_int(merge(n,m,c),kind=8))
! SUBMOD: real(4)::b(1_8:{{.*}}ubound(z,dim=2,kind=8){{.*}})
! SUBMOD-NOT: {{[%][Rr][Oo][Bb][Ee]}}

! No FileCheck: the assertion is that this compiles.  A descendant submodule
! reads m_submod-child.mod from disk and calls sep through it, so this fails
! unless the child's interface body declared z before b.  report() exists only
! to host the call.
!--- submod_grandchild.f90
submodule(m_submod:child) grandchild
contains
  module procedure report
    real :: bb(5), zz(2, 5)
    call sep([2, 5], [7, 7], [.true., .true.], bb, zz)
  end procedure
end submodule

! A written bound must read back as the same expression tree, not merely the
! same value.  The stored extent is max(0,ub) with ub a tagged element printed
! as ubound(z,dim=2,kind=8); re-analyzing that text folds UBOUND back to
! max(0,ub), so without the fold collapsing max(0,max(0,x)) to max(0,x) the
! copy read from the module file would gain a wrapper on every round trip.
! Dummy array shapes are compared leniently and would not notice; function
! results, procedure-pointer targets, and character lengths are compared
! structurally and would be rejected as having distinct extents.  No FileCheck:
! the assertion is that the child and the consumer compile.
!--- fixedpoint_parent.f90
module m_fixedpoint
  interface
    module function g(n, m, c, z) result(r)
      integer, intent(in) :: n(2), m(2)
      logical, intent(in) :: c(2)
      real, intent(in) :: z(merge(n, m, c))
      real :: r(ubound(z, 2))
    end function
    module subroutine s(n, m, c, z, str)
      integer, intent(in) :: n(2), m(2)
      logical, intent(in) :: c(2)
      real, intent(in) :: z(merge(n, m, c))
      character(len=size(z, 2)), intent(in) :: str
    end subroutine
  end interface
end module

! Separate module procedures redeclare the interface's result and character
! length; the comparison against the interface read from m_fixedpoint.mod is
! structural.
!--- fixedpoint_child.f90
submodule(m_fixedpoint) child
contains
  module function g(n, m, c, z) result(r)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real, intent(in) :: z(merge(n, m, c))
    real :: r(ubound(z, 2))
    r = 3.0
  end function
  module subroutine s(n, m, c, z, str)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real, intent(in) :: z(merge(n, m, c))
    character(len=size(z, 2)), intent(in) :: str
  end subroutine
end submodule

! A procedure pointer whose interface is declared locally is associated with
! the module function read back from m_fixedpoint.mod; the result extents are
! compared structurally.
!--- fixedpoint_consumer.f90
program fixedpoint_consumer
  use m_fixedpoint, only: g
  interface
    function gi(n, m, c, z) result(r)
      integer, intent(in) :: n(2), m(2)
      logical, intent(in) :: c(2)
      real, intent(in) :: z(merge(n, m, c))
      real :: r(ubound(z, 2))
    end function
  end interface
  procedure(gi), pointer :: pp
  pp => g
end program

! The reverse orientation: an irreducible rank-1 LOWER bound with a constant
! rank-1 upper bound.  The constant [5,5] is folded into per-dimension constant
! bounds at declaration and must be written back as a rank-1 array -- not as
! the scalar 5 that every dimension happens to share -- so that a separately
! compiled consumer reads d back with rank 2.  The procedure pointer
! association compares the read-back interface structurally against one
! declared locally with the same bounds.
!--- reverse_provider.f90
module m_reverse
contains
  subroutine s(n, m, c, d)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real, intent(inout) :: d(merge(n, m, c) : [5, 5])
    d = 0.0
  end subroutine
end module
! REVERSE-LABEL: subroutine s(n,m,c,d)
! REVERSE: real(4),intent(inout)::d(__builtin_int(merge(n,m,c),kind=8):[INTEGER(8)::5_8,5_8])
! REVERSE-NOT: {{[%][Rr][Oo][Bb][Ee]}}

!--- reverse_consumer.f90
program reverse_consumer
  use m_reverse, only: s
  interface
    subroutine si(n, m, c, d)
      integer, intent(in) :: n(2), m(2)
      logical, intent(in) :: c(2)
      real, intent(inout) :: d(merge(n, m, c) : [5, 5])
    end subroutine
  end interface
  procedure(si), pointer :: pp
  real :: x(2:5, 3:5)
  pp => s
  call s([2, 3], [9, 9], [.true., .true.], x)
end program

!--- consumer.f90
program p
  use m_provenance, only: s1 => s
  use m_unfolded, only: s2 => s
  use m_dependency, only: make_result, s3 => s
  real, allocatable :: x(:)
  x = make_result([2, 3], [4, 5], [.true., .true.])
end program
