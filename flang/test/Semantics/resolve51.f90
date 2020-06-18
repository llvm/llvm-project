! RUN: %S/test_errors.sh %s %t %f18
! Test SELECT TYPE errors: C1157

subroutine s1()
  type :: t
  end type
  procedure(f) :: ff
  !ERROR: Selector is not a named variable: 'associate-name =>' is required
  select type(ff())
    class is(t)
    class default
  end select
contains
  function f()
    class(t), pointer :: f
    f => null()
  end function
end subroutine
