! Code based on the reproducer from https://github.com/flang-compiler/flang/issues/889

module declare_var
  integer, parameter, public :: var = 8
end module declare_var

module declare_var_2
   use declare_var
   integer, parameter, public :: var_2 = var
end module declare_var_2

module declare_type
  ! Removing this use statement and defining var_2 in this module fixes our issue
  use declare_var_2
  type, public :: new_type
  end type new_type
end module declare_type

module declare_func
  use declare_var, only: var_2 => var
  private ! Removing this fixes the issue
  public :: func
  contains
    pure real(var_2) function func()
        func = x'eeeeeeee'
    end function func
end module declare_func

module declare_var_2_again
  implicit none
  private
  integer, parameter, public :: var_2=4 ! This is the value that should be taken as double precision for real(var_2) inside subrout.
end module declare_var_2_again

module mod_subrout
  use declare_var_2_again ! var_2 for real(var_2) should come from this line
  ! Removing this use statement (and use of func) fixes the issue
  use declare_func ! returns var_2=8 defined in declare_var

  integer :: result(3)
  integer :: expect(3)
  data expect/4, 4, 8/ ! kind(unused_var), var_2 and var_3 - see below for more explanations.

  contains
    subroutine subrout(subrout_arg)
      ! Removing this use statement (and use of subrout_arg) fixes the issue
      use declare_type, only: new_type
      ! Removing this use statement with rename fixes it
      use declare_var_2, var_3 => var_2 ! var_2=8 in declare_var_2 so var_3=8
      type(new_type), intent(inout) :: subrout_arg ! Changing to an intrinsic type fixes the issue
      real(var_2) :: unused_var
      ! the value of var_2 should come from declare_var_2_again (equal 4)

      result(1) = kind(unused_var)
      result(2) = var_2
      result(3) = var_3

      call check(result, expect, 3)

    end subroutine subrout
end module mod_subrout

program foo
  use declare_type
  use mod_subrout
  type(new_type) :: t
  call subrout(t)
end program
