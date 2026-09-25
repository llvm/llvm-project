! RUN: %python %S/../test_errors.py %s %flang -fopenmp -Werror
! A DECLARE SIMD directive in an interface body applies to the external
! procedure being declared, not to any definition in this compilation, so
! Flang ignores it. Warn that it has no effect.
! See https://github.com/llvm/llvm-project/issues/192581.

interface
  subroutine add2(i)
  !WARNING: 'DECLARE SIMD' directive in an interface body has no effect [-Wopenmp-usage]
  !$omp declare simd(add2) linear(i:1)
    integer :: i
  end subroutine

  subroutine add3(i)
  !WARNING: 'DECLARE SIMD' directive in an interface body has no effect [-Wopenmp-usage]
  !$omp declare simd
    integer :: i
  end subroutine
end interface

contains

! A DECLARE SIMD in an actual definition is fine and must not warn.
subroutine sub(i)
!$omp declare simd(sub) linear(i:1)
  integer :: i
  i = i + 1
end subroutine
end
