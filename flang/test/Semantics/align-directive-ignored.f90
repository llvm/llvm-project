! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! RUN: %flang_fc1 -fsyntax-only -Werror -Wno-ignored-directive %s
! RUN: %flang_fc1 -fsyntax-only -w %s
! Until lowering honors !DIR$ ALIGN, every recognized directive gets exactly
! one "not yet honored" warning, forms the parser does not recognize keep the
! unrecognized-directive warning, nothing is an error, and both warnings
! belong to -Wignored-directive. A directive outside any program unit is
! handled too (the first and last directives of this file).

!WARNING: !DIR$ ALIGN is recognized but not yet honored by this compiler; the directive is ignored [-Wignored-directive]
!dir$ align(64) :: before_any_unit

subroutine s(n)
  integer :: n
  !WARNING: !DIR$ ALIGN is recognized but not yet honored by this compiler; the directive is ignored [-Wignored-directive]
  !dir$ align(64) :: a, b
  !WARNING: !DIR$ ALIGN is recognized but not yet honored by this compiler; the directive is ignored [-Wignored-directive]
  !dir$ attributes align : 32 :: c
  !WARNING: !DIR$ ALIGN is recognized but not yet honored by this compiler; the directive is ignored [-Wignored-directive]
  !dir$ align(16) d
  ! Forms the parser does not recognize as ALIGN: the COMMON-block forms, an
  ! attribute list, a missing value separator, a named constant, a kind suffix
  ! and a bare ALIGN.
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ attributes align : 64 :: /cb/
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ align(64) /cb/
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ attributes align : 64, dllexport :: a
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ attributes align 64 :: a
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ align(n) :: a
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ align(64_8) :: a
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !dir$ align
  ! !DEC$ lines are plain comments.
  !dec$ attributes align : 64 :: a
  real :: a(n), b(n), c, d(10), e
  common /cb/ e
end subroutine

!WARNING: !DIR$ ALIGN is recognized but not yet honored by this compiler; the directive is ignored [-Wignored-directive]
!dir$ align(64) :: after_last_unit
