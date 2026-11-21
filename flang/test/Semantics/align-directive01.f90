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
  ! The value is a power of two no greater than 2**28.
  !dir$ align(1) :: smallest
  !WARNING: !DIR$ ALIGN on 'smallest' is not yet implemented and was ignored [-Wignored-directive]
  real :: smallest
  !dir$ align(268435456) :: largest
  !WARNING: !DIR$ ALIGN on 'largest' is not yet implemented and was ignored [-Wignored-directive]
  real :: largest
  !ERROR: The alignment in !DIR$ ALIGN must be a power of two no greater than 268435456
  !dir$ align(0) :: zero_value
  !ERROR: The alignment in !DIR$ ALIGN must be a power of two no greater than 268435456
  !dir$ align(24) :: bad_value
  !ERROR: The alignment in !DIR$ ALIGN must be a power of two no greater than 268435456
  !dir$ align(536870912) :: too_big
  real :: zero_value, bad_value, too_big
  !dir$ align(16) :: p
  !ERROR: !DIR$ ALIGN may not apply to named constant 'p'
  real, parameter :: p = 1.0
  ! ALLOCATABLE and POINTER objects are not supported yet: the directive is
  ! ignored with a warning, whichever way the attribute is given.
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'alloc' and was ignored [-Wignored-directive]
  !dir$ align(16) :: alloc
  real, allocatable :: alloc(:)
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'ptr' and was ignored [-Wignored-directive]
  !dir$ align(16) :: ptr
  real, pointer :: ptr(:)
  !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'alloc_stmt' and was ignored [-Wignored-directive]
  !dir$ align(16) :: alloc_stmt
  real :: alloc_stmt(:)
  allocatable :: alloc_stmt
  !dir$ align(16) :: bc
  !ERROR: !DIR$ ALIGN is not yet supported for BIND(C) variable 'bc'
  real, bind(c) :: bc
  ! A coarray is an error even when it is ALLOCATABLE.
  !dir$ align(16) :: co, alloc_co
  !ERROR: !DIR$ ALIGN may not apply to coarray 'co'
  real :: co[*]
  !ERROR: !DIR$ ALIGN may not apply to coarray 'alloc_co'
  real, allocatable :: alloc_co[:]
  !dir$ align(16) :: pointee
  !ERROR: !DIR$ ALIGN may not apply to Cray pointee 'pointee'
  real :: pointee(4)
  pointer (cray_pointer, pointee)
  !dir$ align(32) :: twice
  !ERROR: 'twice' already has a different !DIR$ ALIGN alignment
  !dir$ align(64) :: twice
  !WARNING: !DIR$ ALIGN on 'twice' is not yet implemented and was ignored [-Wignored-directive]
  real :: twice
  !dir$ align(64) :: same_twice
  !dir$ align(64) :: same_twice
  !WARNING: !DIR$ ALIGN on 'same_twice' is not yet implemented and was ignored [-Wignored-directive]
  real :: same_twice
  ! The directive does not declare a name.
  !ERROR: 'not_declared' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
  !dir$ align(64) :: not_declared
  ! Among the components of a derived type, the directive names a component
  ! that is already declared.
  type t
    !ERROR: !DIR$ ALIGN is not supported for component 'comp', which is neither ALLOCATABLE nor POINTER
    real :: comp(8)
    !dir$ align(64) :: comp
    real, allocatable :: acomp(:)
    !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'acomp' and was ignored [-Wignored-directive]
    !dir$ align(64) :: acomp
    !ERROR: 'acomp' already has a different !DIR$ ALIGN alignment
    !dir$ align(32) :: acomp
    !ERROR: !DIR$ ALIGN may not apply to coarray 'ccomp'
    real, allocatable :: ccomp[:]
    !dir$ align(64) :: ccomp
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
    ! A host variable cannot be named, but a local variable that hides one
    ! can, even when its declaration follows the directive.
    !ERROR: 'ok_module' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: ok_module
    !dir$ align(64) :: ok_before
    !WARNING: !DIR$ ALIGN on 'ok_before' is not yet implemented and was ignored [-Wignored-directive]
    real :: ok_before(4)
    !WARNING: !DIR$ ALIGN is ignored for 'c1' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    !dir$ align(64) :: c1
    !WARNING: !DIR$ ALIGN is ignored for 'e1' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    !dir$ align(64) :: e1
    real :: c1, e1, e2
    common /blk/ c1
    equivalence (e1, e2)
    ! A specification statement other than a type declaration declares the
    ! name too.
    !dir$ align(64) :: dim_only
    !WARNING: !DIR$ ALIGN on 'dim_only' is not yet implemented and was ignored [-Wignored-directive]
    dimension dim_only(4)
    ! A name that first appears in the executable part, or only in a NAMELIST
    ! group, is not declared in the specification part.
    !ERROR: 'exec_only' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: exec_only
    !ERROR: 'nml_only' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: nml_only
    namelist /group/ nml_only
    !dir$ align(64) :: h
    !WARNING: !DIR$ ALIGN on 'h' is not yet implemented and was ignored [-Wignored-directive]
    real :: h(10)
    h = 0.0
    exec_only = 0.0
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
    real :: outer
    block
      !WARNING: !DIR$ ALIGN is not yet implemented for ALLOCATABLE or POINTER 'ba' and was ignored [-Wignored-directive]
      !dir$ align(64) :: ba
      real, allocatable :: ba(:)
      !ERROR: 'outer' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
      !dir$ align(64) :: outer
      outer = 1.0
    end block
  end subroutine
  ! An implicitly typed name is declared by any statement of the
  ! specification part that mentions it.
  subroutine implicit_names()
    !dir$ align(64) :: save_only, data_only, volatile_only
    !WARNING: !DIR$ ALIGN on 'save_only' is not yet implemented and was ignored [-Wignored-directive]
    save save_only
    !WARNING: !DIR$ ALIGN on 'data_only' is not yet implemented and was ignored [-Wignored-directive]
    data data_only /1.0/
    !WARNING: !DIR$ ALIGN on 'volatile_only' is not yet implemented and was ignored [-Wignored-directive]
    volatile volatile_only
    !WARNING: !DIR$ ALIGN is ignored for 'common_only' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    !dir$ align(64) :: common_only
    !WARNING: !DIR$ ALIGN is ignored for 'equivalence_only' because it is in COMMON or EQUIVALENCE storage [-Wignored-directive]
    !dir$ align(64) :: equivalence_only
    common /implicit_blk/ common_only
    equivalence (equivalence_only, other)
  end subroutine
  subroutine implicit_rule()
    implicit integer(a-z)
    !dir$ align(64) :: by_rule
    !WARNING: !DIR$ ALIGN on 'by_rule' is not yet implemented and was ignored [-Wignored-directive]
    dimension by_rule(4)
  end subroutine
  subroutine no_implicit()
    implicit none
    !ERROR: 'undeclared' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: undeclared
  end subroutine
  ! With a host variable of the same name, DIMENSION and SAVE declare a
  ! local variable, and VOLATILE refers to the host variable.
  subroutine host_names()
    !ERROR: 'ok_intel' must be declared in this scoping unit to be aligned with !DIR$ ALIGN
    !dir$ align(64) :: ok_module, ok_after, ok_intel
    !WARNING: !DIR$ ALIGN on 'ok_module' is not yet implemented and was ignored [-Wignored-directive]
    dimension ok_module(4)
    !WARNING: !DIR$ ALIGN on 'ok_after' is not yet implemented and was ignored [-Wignored-directive]
    save ok_after
    volatile ok_intel
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
