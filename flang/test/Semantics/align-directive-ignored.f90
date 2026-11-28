! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! RUN: %flang_fc1 -fsyntax-only -Werror -Wno-ignored-directive %s
! RUN: %flang_fc1 -fsyntax-only -w %s
! Until lowering honors !DIR$ ALIGN, every object named by an accepted
! directive gets exactly one "not yet implemented" warning, forms the parser
! does not recognize as ALIGN keep the unrecognized-directive warning, nothing
! is an error, and both warnings belong to -Wignored-directive.

subroutine s(n)
  integer :: n
  !dir$ align(64) :: a, b
  !dir$ attributes align : 32 :: c
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
  !WARNING: !DIR$ ALIGN on 'a' is not yet implemented and was ignored [-Wignored-directive]
  !WARNING: !DIR$ ALIGN on 'b' is not yet implemented and was ignored [-Wignored-directive]
  !WARNING: !DIR$ ALIGN on 'c' is not yet implemented and was ignored [-Wignored-directive]
  !WARNING: !DIR$ ALIGN on 'd' is not yet implemented and was ignored [-Wignored-directive]
  real :: a(n), b(n), c, d(10), e
  common /cb/ e
end subroutine
