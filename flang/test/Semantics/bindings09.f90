! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Ensure that generic resolution for a parameterized derived type uses the
! module containing the selected binding, not the PDT instantiation site.

module private_pdt
  implicit none
  integer, parameter :: sp = kind(1.0)

  type, abstract :: base_t(k)
    integer, kind :: k = sp
  contains
    procedure(private_interface), private, deferred :: binding
    generic, public :: generic => binding
  end type

  type, extends(base_t) :: extension_t
  contains
    procedure, private :: binding => private_impl
  end type

  abstract interface
    subroutine private_interface(x, n)
      import base_t, sp
      class(base_t(sp)), intent(inout) :: x
      integer, intent(in) :: n
    end subroutine
  end interface

  type(extension_t(sp)), public :: module_object

contains
  subroutine private_impl(x, n)
    class(extension_t(sp)), intent(inout) :: x
    integer, intent(in) :: n
  end subroutine
end module

module third_module
  use private_pdt
  implicit none
contains
  subroutine call_from_third_module
    type(extension_t(sp)) :: x
    ! CHECK: CALL private_impl(x,1_4)
    call x%generic(1)
  end subroutine
end module

module use_renamed
  use private_pdt, only: renamed_t => extension_t, sp
  implicit none
contains
  subroutine call_use_renamed
    type(renamed_t(sp)) :: x
    ! CHECK: CALL private_impl(x,2_4)
    call x%generic(2)
  end subroutine
end module

module private_cross_module
  use private_pdt
  implicit none

  ! The inherited private binding cannot be overridden in another module.
  type, extends(extension_t) :: further_extension_t
  contains
    procedure :: binding => unrelated_impl
  end type

contains
  subroutine unrelated_impl(x, n)
    class(further_extension_t(sp)), intent(inout) :: x
    integer, intent(in) :: n
  end subroutine

  subroutine call_private_cross_module
    type(further_extension_t(sp)) :: x
    ! CHECK: CALL private_impl(x,3_4)
    call x%generic(3)
  end subroutine
end module

module public_pdt
  implicit none
  integer, parameter :: sp = kind(1.0)

  type :: base_t(k)
    integer, kind :: k = sp
  contains
    procedure, public :: binding => public_base_impl
    generic, public :: generic => binding
  end type

contains
  subroutine public_base_impl(x, n)
    class(base_t(sp)), intent(inout) :: x
    integer, intent(in) :: n
  end subroutine
end module

module public_cross_module
  use public_pdt
  implicit none

  ! Unlike a private binding, the public binding remains overridable here.
  type, extends(base_t) :: extension_t
  contains
    procedure :: binding => public_extension_impl
  end type

contains
  subroutine public_extension_impl(x, n)
    class(extension_t(sp)), intent(inout) :: x
    integer, intent(in) :: n
  end subroutine

  subroutine call_public_cross_module
    type(extension_t(sp)) :: x
    ! CHECK: CALL public_extension_impl(x,4_4)
    call x%generic(4)
  end subroutine
end module

program test
  use private_pdt
  implicit none
  type(extension_t(sp)) :: x

  ! CHECK: CALL private_impl(x,5_4)
  call x%generic(5)
  ! CHECK: CALL private_impl(module_object,6_4)
  call module_object%generic(6)
end program
