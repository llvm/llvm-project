! RUN: %flang_fc1 -fsyntax-only %s
! Nonconstant operands exercise the NOT folding fallback without collapsing
! the nested expression tree to a constant. This is a compile-time smoke test;
! it does not impose a timing threshold.
! For a manual before/after benchmark, set FLANG_BEFORE and FLANG_AFTER to
! equivalently configured compiler builds and run from the same checkout:
!   chain_test=flang/test/Evaluate/nonconstant-not-chain.F90
!   time "$FLANG_BEFORE" -fc1 -fsyntax-only "$chain_test"
!   time "$FLANG_AFTER" -fc1 -fsyntax-only "$chain_test"
! Repeat the commands to account for startup, filesystem, and host-load effects.
#define N1(x) .not.(x)
#define N2(x) N1(N1(x))
#define N4(x) N2(N2(x))
#define N8(x) N4(N4(x))
#define N16(x) N8(N8(x))
#define N32(x) N16(N16(x))
#define N64(x) N32(N32(x))
#define N128(x) N64(N64(x))
#define N256(x) N128(N128(x))

subroutine not_scalar_chain(x, y)
  logical, intent(in) :: x
  logical, intent(out) :: y
  y = N256(x)
end subroutine

subroutine not_array_chain(x, y)
  logical(8), intent(in) :: x(:)
  logical(8), intent(out) :: y(size(x))
  y = .not. N256(x)
end subroutine
