! RUN: %S/test_errors.sh %s %t %f18

module m1
  implicit none
  type t
    integer :: n
  end type
  type t2
    ! t and t2 must be resolved to types in m, not components in t2
    type(t) :: t(10) = t(1)
    type(t) :: x = t(1)
    integer :: t2
    type(t2), pointer :: p
  end type
end

module m2
  type :: t(t)
    integer, kind :: t
    integer(t) :: n
  end type
  type :: t2(t)
    integer, kind :: t
    type(t(t)) :: x = t(t)(t)
  end type
end
