! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C1140 -- A statement that might result in the deallocation of a polymorphic 
! entity shall not appear within a DO CONCURRENT construct.
module m1
  ! Base type with scalar components
  type :: Base
    integer :: baseField1
  end type

  ! Child type so we can allocate polymorphic entities
  type, extends(Base) :: ChildType
    integer :: childField
  end type

  ! Type with a polymorphic, allocatable component
  type, extends(Base) :: HasAllocPolyType
    class(Base), allocatable :: allocPolyField
  end type

  ! Type with a allocatable, coarray component
  type :: HasAllocCoarrayType
    type(Base), allocatable, codimension[:] :: allocCoarrayField
  end type

  ! Type with a polymorphic, allocatable, coarray component
  type :: HasAllocPolyCoarrayType
    class(Base), allocatable, codimension[:] :: allocPolyCoarrayField
  end type

  ! Type with a polymorphic, pointer component
  type, extends(Base) :: HasPointerPolyType
    class(Base), pointer :: pointerPolyField
  end type

  class(Base), allocatable :: baseVar1
  type(Base) :: baseVar2
end module m1

subroutine s1()
  ! Test deallocation of polymorphic entities caused by block exit
  use m1

  block
    ! The following should not cause problems
    integer :: outerInt

    ! The following are OK since they're not in a DO CONCURRENT
    class(Base), allocatable :: outerAllocatablePolyVar
    class(Base), allocatable, codimension[:] :: outerAllocatablePolyCoarray
    type(HasAllocPolyType), allocatable  :: outerAllocatableWithAllocPoly
    type(HasAllocPolyCoarrayType), allocatable :: outerAllocWithAllocPolyCoarray

    do concurrent (i = 1:10)
      ! The following should not cause problems
      block
        integer, allocatable :: blockInt
      end block
      block
        ! Test polymorphic entities
        ! OK because it's a pointer to a polymorphic entity
        class(Base), pointer :: pointerPoly

        ! OK because it's not polymorphic
        integer, allocatable :: intAllocatable

        ! OK because it's not polymorphic
        type(Base), allocatable :: allocatableNonPolyBlockVar

        ! Bad because it's polymorphic and allocatable
        class(Base), allocatable :: allocatablePoly

        ! OK because it has the SAVE attribute
        class(Base), allocatable, save :: allocatablePolySave

        ! Bad because it's polymorphic and allocatable
        class(Base), allocatable, codimension[:] :: allocatablePolyCoarray

        ! OK because it's not polymorphic and allocatable
        type(Base), allocatable, codimension[:] :: allocatableCoarray

        ! Bad because it has a allocatable polymorphic component
        type(HasAllocPolyType), allocatable  :: allocatableWithAllocPoly

        ! OK because the declared variable is not allocatable
        type(HasAllocPolyType) :: nonAllocatableWithAllocPoly

        ! OK because the declared variable is not allocatable
        type(HasAllocPolyCoarrayType) :: nonAllocatableWithAllocPolyCoarray

        ! Bad because even though the declared the allocatable component is a coarray
        type(HasAllocPolyCoarrayType), allocatable :: allocWithAllocPolyCoarray

        ! OK since it has no polymorphic component
        type(HasAllocCoarrayType) :: nonAllocWithAllocCoarray

        ! OK since it has no component that's polymorphic, oops
        type(HasPointerPolyType), allocatable :: allocatableWithPointerPoly

!ERROR: Deallocation of a polymorphic entity caused by block exit not allowed in DO CONCURRENT
!ERROR: Deallocation of a polymorphic entity caused by block exit not allowed in DO CONCURRENT
!ERROR: Deallocation of a polymorphic entity caused by block exit not allowed in DO CONCURRENT
!ERROR: Deallocation of a polymorphic entity caused by block exit not allowed in DO CONCURRENT
      end block
    end do
  end block

end subroutine s1

subroutine s2()
  ! Test deallocation of a polymorphic entity cause by intrinsic assignment
  use m1

  class(Base), allocatable :: localVar
  class(Base), allocatable :: localVar1
  type(Base), allocatable :: localVar2

  type(HasAllocPolyType), allocatable :: polyComponentVar
  type(HasAllocPolyType), allocatable :: polyComponentVar1

  type(HasAllocPolyType) :: nonAllocPolyComponentVar
  type(HasAllocPolyType) :: nonAllocPolyComponentVar1
  class(HasAllocPolyCoarrayType), allocatable :: allocPolyCoarray
  class(HasAllocPolyCoarrayType), allocatable :: allocPolyCoarray1

  class(Base), allocatable, codimension[:] :: allocPolyComponentVar
  class(Base), allocatable, codimension[:] :: allocPolyComponentVar1

  allocate(ChildType :: localVar)
  allocate(ChildType :: localVar1)
  allocate(Base :: localVar2)
  allocate(polyComponentVar)
  allocate(polyComponentVar1)
  allocate(allocPolyCoarray)
  allocate(allocPolyCoarray1)

  ! These are OK because they're not in a DO CONCURRENT
  localVar = localVar1
  nonAllocPolyComponentVar = nonAllocPolyComponentVar1
  polyComponentVar = polyComponentVar1
  allocPolyCoarray = allocPolyCoarray1

  do concurrent (i = 1:10)
    ! Test polymorphic entities
    ! Bad because localVar is allocatable and polymorphic, 10.2.1.3, par. 3
!ERROR: Deallocation of a polymorphic entity caused by assignment not allowed in DO CONCURRENT
    localVar = localVar1

    ! The next one should be OK since localVar2 is not polymorphic
    localVar2 = localVar1

    ! Bad because the copying of the components causes deallocation
!ERROR: Deallocation of a polymorphic entity caused by assignment not allowed in DO CONCURRENT
    nonAllocPolyComponentVar = nonAllocPolyComponentVar1

    ! Bad because possible deallocation a variable with a polymorphic component
!ERROR: Deallocation of a polymorphic entity caused by assignment not allowed in DO CONCURRENT
    polyComponentVar = polyComponentVar1

    ! Bad because deallocation upon assignment happens with allocatable
    ! entities, even if they're coarrays.  The noncoarray restriction only
    ! applies to components
!ERROR: Deallocation of a polymorphic entity caused by assignment not allowed in DO CONCURRENT
    allocPolyCoarray = allocPolyCoarray1

  end do
end subroutine s2

subroutine s3()
  ! Test direct deallocation
  use m1

  class(Base), allocatable :: polyVar
  type(Base), allocatable :: nonPolyVar
  type(HasAllocPolyType), allocatable :: polyComponentVar
  type(HasAllocPolyType), pointer :: pointerPolyComponentVar

  allocate(ChildType:: polyVar)
  allocate(nonPolyVar)
  allocate(polyComponentVar)
  allocate(pointerPolyComponentVar)

  ! These are all good because they're not in a do concurrent
  deallocate(polyVar)
  allocate(polyVar)
  deallocate(polyComponentVar)
  allocate(polyComponentVar)
  deallocate(pointerPolyComponentVar)
  allocate(pointerPolyComponentVar)

  do concurrent (i = 1:10)
    ! Bad because deallocation of a polymorphic entity
!ERROR: Deallocation of a polymorphic entity caused by a DEALLOCATE statement not allowed in DO CONCURRENT
    deallocate(polyVar)

    ! Bad, deallocation of an entity with a polymorphic component
!ERROR: Deallocation of a polymorphic entity caused by a DEALLOCATE statement not allowed in DO CONCURRENT
    deallocate(polyComponentVar)

    ! Bad, deallocation of a pointer to an entity with a polymorphic component
!ERROR: Deallocation of a polymorphic entity caused by a DEALLOCATE statement not allowed in DO CONCURRENT
    deallocate(pointerPolyComponentVar)

    ! Deallocation of a nonpolymorphic entity
    deallocate(nonPolyVar)
  end do
end subroutine s3

module m2
  type :: impureFinal
   contains
    final :: impureSub
  end type

  type :: pureFinal
   contains
    final :: pureSub
  end type

 contains

  impure subroutine impureSub(x)
    type(impureFinal), intent(in) :: x
  end subroutine

  pure subroutine pureSub(x)
    type(pureFinal), intent(in) :: x
  end subroutine

  subroutine s4()
    type(impureFinal), allocatable :: ifVar, ifvar1
    type(pureFinal), allocatable :: pfVar
    allocate(ifVar)
    allocate(ifVar1)
    allocate(pfVar)

    ! OK for an ordinary DO loop
    do i = 1,10
      if (i .eq. 1) deallocate(ifVar)
    end do

    ! OK to invoke a PURE FINAL procedure in a DO CONCURRENT
    ! This case does not work currently because the compiler's test for
    ! HasImpureFinal() in .../lib/Semantics/tools.cc doesn't work correctly
!    do concurrent (i = 1:10)
!      if (i .eq. 1) deallocate(pfVar)
!    end do

    ! Error to invoke an IMPURE FINAL procedure in a DO CONCURRENT
    do concurrent (i = 1:10)
          !ERROR: Deallocation of an entity with an IMPURE FINAL procedure caused by a DEALLOCATE statement not allowed in DO CONCURRENT
      if (i .eq. 1) deallocate(ifVar)
    end do

    do concurrent (i = 1:10)
      if (i .eq. 1) then
        block
          type(impureFinal), allocatable :: ifVar
          allocate(ifVar)
          ! Error here because exiting this scope causes the finalization of 
          !ifvar which causes the invocation of an IMPURE FINAL procedure
          !ERROR: Deallocation of an entity with an IMPURE FINAL procedure caused by block exit not allowed in DO CONCURRENT
        end block
      end if
    end do

    do concurrent (i = 1:10)
      if (i .eq. 1) then
        ! Error here because the assignment statement causes the finalization 
        ! of ifvar which causes the invocation of an IMPURE FINAL procedure
!ERROR: Deallocation of an entity with an IMPURE FINAL procedure caused by assignment not allowed in DO CONCURRENT
        ifvar = ifvar1
      end if
    end do
  end subroutine s4

end module m2
