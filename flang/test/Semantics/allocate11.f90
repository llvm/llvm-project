! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

! TODO: Function Pointer in allocate and derived types!

! Rules I should know when working with coarrays and derived type:

! C736: If EXTENDS appears and the type being defined has a coarray ultimate
! component, its parent type shall have a coarray ultimate component.

! C746: (R737) If a coarray-spec appears, it shall be a deferred-coshape-spec-list
! and the component shall have the ALLOCATABLE attribute.

! C747: If a coarray-spec appears, the component shall not be of type C_PTR or
! C_FUNPTR from the intrinsic module ISO_C_BINDING (18.2), or of type TEAM_TYPE from the
! intrinsic module ISO_FORTRAN_ENV (16.10.2).

! C748: A data component whose type has a coarray ultimate component shall be a
! nonpointer nonallocatable scalar and shall not be a coarray.

! 7.5.4.3 Coarray components
! 7.5.6 Final subroutines: C786


! C825 An entity whose type has a coarray ultimate component shall be a
! nonpointer nonallocatable scalar, shall not be a coarray, and shall not be a function result.

! C826 A coarray or an object with a coarray ultimate component shall be an
! associate name, a dummy argument, or have the ALLOCATABLE or SAVE attribute.

subroutine C937(var)
! Type-spec shall not specify a type that has a coarray ultimate component.


  type A
    real, allocatable :: x[:]
  end type

  type B
    type(A) y
    !ERROR: A component with a POINTER or ALLOCATABLE attribute may not be of a type with a coarray ultimate component (named 'y%x')
    type(B), pointer :: forward
    real :: u
  end type

  type C
    type(B) z
  end type

  type D
    !ERROR: A component with a POINTER or ALLOCATABLE attribute may not be of a type with a coarray ultimate component (named 'x')
    type(A), pointer :: potential
  end type



  class(*), allocatable :: var
  ! unlimited polymorphic is the ONLY way to get an allocatable/pointer 'var' that can be
  ! allocated with a type-spec T that has coarray ultimate component without
  ! violating other rules than C937.
  ! Rationale:
  !   C934 => var must be type compatible with T.
  !        => var type is T, a type P extended by T, or unlimited polymorphic
  !   C825 => var cannot be of type T.
  !   C736 => all parent types P of T must have a coarray ultimate component
  !        => var cannot be of type P (C825)
  !        => if var can be defined, it can only be unlimited polymorphic

  ! Also, as per C826 or C852, var can only be an allocatable, not a pointer

  ! OK, x is not an ultimate component
  allocate(D:: var)

  !ERROR: Type-spec in ALLOCATE must not specify a type with a coarray ultimate component
  allocate(A:: var)
  !ERROR: Type-spec in ALLOCATE must not specify a type with a coarray ultimate component
  allocate(B:: var)
  !ERROR: Type-spec in ALLOCATE must not specify a type with a coarray ultimate component
  allocate(C:: var)
end subroutine

!TODO: type extending team_type !? subcomponents !?

subroutine C938_C947(var2, ptr, ptr2, fptr, my_team, srca)
! If an allocate-object is a coarray, type-spec shall not specify type C_PTR or
! C_FUNPTR from the intrinsic module ISO_C_BINDING, or type TEAM_TYPE from the intrinsic module
! ISO_FORTRAN_ENV.
  use ISO_FORTRAN_ENV
  use ISO_C_BINDING

  type A(k, l)
    integer, kind :: k
    integer, len :: l
    real(kind=k) x(l,l)
  end type

! Again, I do not see any other way to violate this rule and not others without
! having var being an unlimited polymorphic.
! Suppose var of type P and T, the type in type-spec
! Per C934, P must be compatible with T. P cannot be a forbidden type per C824.
! Per C728 and 7.5.7.1, P cannot extend a c_ptr or _c_funptr. hence, P has to be
! unlimited polymorphic or a type that extends TEAM_TYPE.
  class(*), allocatable :: var[:], var2(:)[:]
  class(*), allocatable :: varok, varok2(:)

  Type(C_PTR) :: ptr, ptr2(2:10)
  Type(C_FUNPTR) fptr
  Type(TEAM_TYPE) my_team
  Type(A(4, 10)) :: srca

  ! Valid constructs
  allocate(real:: var[5:*])
  allocate(A(4, 10):: var[5:*])
  allocate(TEAM_TYPE:: varok, varok2(2))
  allocate(C_PTR:: varok, varok2(2))
  allocate(C_FUNPTR:: varok, varok2(2))

  !ERROR: Type-Spec in ALLOCATE must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray
  allocate(TEAM_TYPE:: var[5:*])
  !ERROR: Type-Spec in ALLOCATE must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(C_PTR:: varok, var[5:*])
  !ERROR: Type-Spec in ALLOCATE must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(C_FUNPTR:: var[5:*])
  !ERROR: Type-Spec in ALLOCATE must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray
  allocate(TEAM_TYPE:: var2(2)[5:*])
  !ERROR: Type-Spec in ALLOCATE must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(C_PTR:: var2(2)[5:*])
  !ERROR: Type-Spec in ALLOCATE must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(C_FUNPTR:: varok2(2), var2(2)[5:*])


! C947: The declared type of source-expr shall not be C_PTR or C_FUNPTR from the
! intrinsic module ISO_C_BINDING, or TEAM_TYPE from the intrinsic module
! ISO_FORTRAN_ENV, if an allocateobject is a coarray.
!
!  ! Valid constructs
  allocate(var[5:*], SOURCE=cos(0.5_4))
  allocate(var[5:*], MOLD=srca)
  allocate(varok, varok2(2), SOURCE=ptr)
  allocate(varok2, MOLD=ptr2)
  allocate(varok, varok2(2), SOURCE=my_team)
  allocate(varok, varok2(2), MOLD=fptr)

  !ERROR: SOURCE or MOLD expression type must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray
  allocate(var[5:*], SOURCE=my_team)
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(var[5:*], SOURCE=ptr)
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(varok, var[5:*], MOLD=ptr2(1))
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(var[5:*], MOLD=fptr)
  !ERROR: SOURCE or MOLD expression type must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray
  allocate(var2(2)[5:*], MOLD=my_team)
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(var2(2)[5:*], MOLD=ptr)
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(var2(2)[5:*], SOURCE=ptr2)
  !ERROR: SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray
  allocate(varok2(2), var2(2)[5:*], SOURCE=fptr)

end subroutine
