! RUN: %python %S/test_errors.py %s %flang_fc1 -fno-realloc-lhs -Werror

! When -fno-realloc-lhs is specified, assignments to a polymorphic allocatable
! LHS must still use reallocation semantics (F2003 requirement).  A warning is
! issued so the user knows the option is being ignored for this case.

subroutine test_unlimited_poly(x)
  class(*), allocatable :: x
! WARNING: -fno-realloc-lhs is ignored for assignment to polymorphic allocatable [-Wignored-no-reallocate-lhs]
  x = 1
end subroutine

subroutine test_class_poly(x)
  type :: t
    integer :: i
  end type
  class(t), allocatable :: x
! WARNING: -fno-realloc-lhs is ignored for assignment to polymorphic allocatable [-Wignored-no-reallocate-lhs]
  x = t(42)
end subroutine

! Non-polymorphic allocatable: no warning, option is honoured.
subroutine test_non_poly(x)
  integer, allocatable :: x
  x = 1
end subroutine
