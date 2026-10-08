! RUN: %python %S/test_errors.py %s %flang_fc1
! F2018 C1595: a procedure referenced from a pure subprogram must be pure,
! including references through defined operators. The operator's generic
! is declared in the module; the check must use the use site.

module pure_op_mod
  implicit none
  type, public :: pv_type
    real :: x = 0.0, y = 0.0
  end type pv_type

  interface operator(+)
    module function pv_add(a, b) result(r)
      type(pv_type), intent(in) :: a, b
      type(pv_type) :: r
    end function pv_add
  end interface operator(+)

  interface operator(*)
    module function pv_scale(s, v) result(r)
      real, intent(in) :: s
      type(pv_type), intent(in) :: v
      type(pv_type) :: r
    end function pv_scale
  end interface operator(*)

  interface operator(-)
    pure module function pv_sub(a, b) result(r)
      type(pv_type), intent(in) :: a, b
      type(pv_type) :: r
    end function pv_sub
  end interface operator(-)
end module pure_op_mod

submodule (pure_op_mod) pure_op_smod
  implicit none
contains
  module procedure pv_add
    r%x = a%x + b%x
    r%y = a%y + b%y
  end procedure pv_add

  module procedure pv_scale
    r%x = s * v%x
    r%y = s * v%y
  end procedure pv_scale

  module procedure pv_sub
    r%x = a%x - b%x
    r%y = a%y - b%y
  end procedure pv_sub
end submodule pure_op_smod

program main
  use pure_op_mod
  implicit none
  type(pv_type) :: a, b, m
  a = pv_type(1.0, 2.0)
  b = pv_type(3.0, 4.0)
  ! Impure defined operators are fine outside a pure subprogram.
  m = 0.5 * (a + b)
  m = a - b
  m = midpoint(a, b)
contains
  pure function midpoint(p, q) result(r)
    type(pv_type), intent(in) :: p, q
    type(pv_type) :: r
    ! A pure defined operator may be referenced.
    r = p - q
    !ERROR: Procedure 'pv_add' referenced in pure subprogram 'midpoint' must be pure too
    !ERROR: Procedure 'pv_scale' referenced in pure subprogram 'midpoint' must be pure too
    r = 0.5 * (p + q)
  end function midpoint
end program main
