! RUN: %python %S/test_errors.py %s %flang_fc1
! !DIR$ ALIGN: placement, value and applicability checks.

!ERROR: !DIR$ ALIGN directive must appear in a program unit
!dir$ align(64) :: x

module m
  !dir$ align(64) :: ok_module
  !WARNING: !DIR$ ALIGN on 'ok_module' is not yet implemented and was ignored [-Wignored-directive]
  real :: ok_module(10)
  !dir$ align(32) :: ok_before, ok_after
  !WARNING: !DIR$ ALIGN on 'ok_before' is not yet implemented and was ignored [-Wignored-directive]
  real :: ok_before
  !WARNING: !DIR$ ALIGN on 'ok_after' is not yet implemented and was ignored [-Wignored-directive]
  real :: ok_after
  !dir$ attributes align : 16 :: ok_intel
  !WARNING: !DIR$ ALIGN on 'ok_intel' is not yet implemented and was ignored [-Wignored-directive]
  real :: ok_intel(4)
  !ERROR: The alignment in !DIR$ ALIGN must be a power of two no greater than 268435456
  !dir$ align(24) :: bad_value
  real :: bad_value
  !ERROR: The alignment in !DIR$ ALIGN must be a power of two no greater than 268435456
  !dir$ align(536870912) :: too_big
  real :: too_big
  !dir$ align(16) :: p
  !ERROR: !DIR$ ALIGN may not apply to named constant 'p'
  real, parameter :: p = 1.0
  ! ALLOCATABLE and POINTER objects are not supported yet: the directive is
  ! ignored with a warning, whichever way the attribute is given.
  !dir$ align(16) :: alloc
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'alloc' and was ignored [-Wignored-directive]
  real, allocatable :: alloc(:)
  !dir$ align(16) :: ptr
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'ptr' and was ignored [-Wignored-directive]
  real, pointer :: ptr(:)
  !dir$ align(16) :: alloc_stmt
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'alloc_stmt' and was ignored [-Wignored-directive]
  real :: alloc_stmt(:)
  allocatable :: alloc_stmt
  !dir$ align(16) :: bc
  !ERROR: !DIR$ ALIGN is not yet supported for BIND(C) variable 'bc'
  real, bind(c) :: bc
  !dir$ align(16) :: co
  !ERROR: !DIR$ ALIGN may not apply to coarray 'co'
  real :: co[*]
  !dir$ align(32) :: twice
  !ERROR: 'twice' already has a different !DIR$ ALIGN alignment
  !dir$ align(64) :: twice
  !WARNING: !DIR$ ALIGN on 'twice' is not yet implemented and was ignored [-Wignored-directive]
  real :: twice
  !dir$ align(64) :: same_twice
  !dir$ align(64) :: same_twice
  !WARNING: !DIR$ ALIGN on 'same_twice' is not yet implemented and was ignored [-Wignored-directive]
  real :: same_twice
  ! Among the components of a derived type, the directive names a component
  ! that is already declared; no component is supported yet.
  type t
    !ERROR: !DIR$ ALIGN is not supported for component 'comp', which is neither ALLOCATABLE nor POINTER
    real :: comp(8)
    !dir$ align(64) :: comp
    real, allocatable :: acomp(:)
    !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'acomp' and was ignored [-Wignored-directive]
    !dir$ align(64) :: acomp
    !ERROR: 'later' must be a previously declared component of this derived type to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: later
    real :: later
    !ERROR: 'ok_module' must be a previously declared component of this derived type to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: ok_module
    procedure(), pointer, nopass :: pcomp
    !ERROR: 'pcomp' must be a variable to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: pcomp
  end type
  !ERROR: 'sub' must be a variable to be aligned with !DIR$ ALIGN
  !dir$ align(64) :: sub
contains
  subroutine sub(d, ad)
    !dir$ align(64) :: d
    !ERROR: !DIR$ ALIGN may not apply to dummy argument 'd'
    real :: d(10)
    !dir$ align(64) :: ad
    !ERROR: !DIR$ ALIGN may not apply to dummy argument 'ad'
    real, allocatable :: ad(:)
    !ERROR: 'ok_module' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: ok_module
    !dir$ align(64) :: c1
    !dir$ align(64) :: e1
    !WARNING: !DIR$ ALIGN is ignored for 'c1' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    !WARNING: !DIR$ ALIGN is ignored for 'e1' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    real :: c1, e1, e2
    common /blk/ c1
    equivalence (e1, e2)
    !dir$ align(64) :: h
    !WARNING: !DIR$ ALIGN on 'h' is not yet implemented and was ignored [-Wignored-directive]
    real :: h(10)
    h = 0.0
    !ERROR: !DIR$ ALIGN directive must appear in the specification part
    !dir$ align(64) :: h
  contains
    subroutine inner()
      !ERROR: 'h' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
      !dir$ align(64) :: h
      h = 1.0
    end subroutine
  end subroutine
  subroutine in_block()
    block
      !dir$ align(64) :: ba
      !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'ba' and was ignored [-Wignored-directive]
      real, allocatable :: ba(:)
    end block
  end subroutine
  function f()
    !dir$ align(64) :: f
    !ERROR: !DIR$ ALIGN may not apply to function result 'f'
    real :: f(10)
    f = 0.0
  end function
end module

! A directive after the last program unit is parsed as a trailing part of
! that unit, outside its specification part.
!ERROR: !DIR$ ALIGN directive must appear in the specification part
!dir$ align(64) :: y
