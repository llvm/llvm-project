!===-- module/__fortran_type_info.f90 --------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! Fortran definitions of runtime type description schemata.
! See flang/runtime/type-info.h for C++ perspective.
! The Semantics phase of the compiler requires the module file of this module
! in order to generate description tables for all other derived types.

module __Fortran_type_info

  private

  integer, parameter :: int64 = selected_int_kind(18)

  type, public :: __builtin_c_ptr
    integer(kind=int64) :: __address
  end type

  type, public :: __builtin_c_funptr
    integer(kind=int64) :: __address
  end type

  type :: DerivedType
    ! "TBP" bindings appear first.  Inherited bindings, with overrides already
    ! applied, appear in the initial entries in the same order as they
    ! appear in the parent type's bindings, if any.  They are followed
    ! by new local bindings in alphabetic order of theing binding names.
    type(Binding), pointer, contiguous :: binding(:)
    character(len=:), pointer :: name
    integer(kind=int64) :: sizeInBytes
    type(DerivedType), pointer :: parent
    ! Instances of parameterized derived types use the "uninstantiated"
    ! component to point to the pristine original definition.
    type(DerivedType), pointer :: uninstantiated
    integer(kind=int64) :: typeHash
    integer(kind=int64), pointer, contiguous :: kindParameter(:) ! values of instance
    integer(1), pointer, contiguous :: lenParameterKind(:) ! INTEGER kinds of LEN types
    ! Data components appear in alphabetic order.
    ! The parent component, if any, appears explicitly.
    type(Component), pointer, contiguous :: component(:) ! data components
    type(ProcPtrComponent), pointer, contiguous :: procptr(:) ! procedure pointers
    ! Special bindings of the ancestral types are not duplicated here.
    type(SpecialBinding), pointer, contiguous :: special(:)
  end type

  type :: Binding
    type(__builtin_c_funptr) :: proc
    character(len=:), pointer :: name
  end type

  ! Array bounds and type parameters of components are deferred
  ! (for allocatables and pointers), explicit constants, or
  ! taken from LEN type parameters for automatic components.
  enum, bind(c) ! Value::Genre
    enumerator :: Deferred = 1, Explicit = 2, LenParameter = 3
  end enum

  type, bind(c) :: Value
    integer(1) :: genre ! Value::Genre
    integer(1) :: __padding0(7)
    integer(kind=int64) :: value
  end type

  enum, bind(c) ! Component::Genre
    enumerator :: Data = 1, Pointer = 2, Allocatable = 3, Automatic = 4
  end enum

  enum, bind(c) ! common::TypeCategory
    enumerator :: CategoryInteger = 0, CategoryReal = 1, &
      CategoryComplex = 2, CategoryCharacter = 3, &
      CategoryLogical = 4, CategoryDerived = 5
  end enum

  type :: Component ! data components, incl. object pointers
    character(len=:), pointer :: name
    integer(1) :: genre ! Component::Genre
    integer(1) :: category
    integer(1) :: kind
    integer(1) :: rank
    integer(1) :: __padding0(4)
    integer(kind=int64) :: offset
    type(Value) :: characterLen ! for category == Character
    type(DerivedType), pointer :: derived ! for category == Derived
    type(Value), pointer, contiguous :: lenValue(:) ! (SIZE(derived%lenParameterKind))
    type(Value), pointer, contiguous :: bounds(:, :) ! (2, rank): lower, upper
    type(__builtin_c_ptr) :: initialization
  end type

  type :: ProcPtrComponent ! procedure pointer components
    character(len=:), pointer :: name
    integer(kind=int64) :: offset
    type(__builtin_c_funptr) :: initialization
  end type

  enum, bind(c) ! SpecialBinding::Which
    enumerator :: Assignment = 4, ElementalAssignment = 5
    enumerator :: Final = 8, ElementalFinal = 9, AssumedRankFinal = 10
    enumerator :: ReadFormatted = 16, ReadUnformatted = 17
    enumerator :: WriteFormatted = 18, WriteUnformatted = 19
  end enum

  type, bind(c) :: SpecialBinding
    integer(1) :: which ! SpecialBinding::Which
    integer(1) :: rank ! for which == SpecialBinding::Which::Final only
    integer(1) :: isArgDescriptorSet
    integer(1) :: __padding0(5)
    type(__builtin_c_funptr) :: proc
  end type

end module
